"""
Extract and expose GPMF data.

"""

from __future__ import annotations

import dataclasses
import itertools
import math
import pathlib
import sys
from typing import Any, Self, Sequence, cast

import numpy as np
import scipy

from .gpmf import Data, Record, parse_gpmf
from .mp4 import MP4File


__all__ = ["Database"]


Metadata = dict[str, Any]


class Stream:
    def __init__(self, fourcc: str, records: Sequence[Record], metadata: Metadata):
        self.fourcc = fourcc
        self.records = records
        self.metadata = metadata
        # Initialisation requires:
        # * self.fit_timestamps() to set self.num_samples;
        # * self.set_timings() to set self.sample_duration,
        #   self.start_timestamp, and self.total_duration.
        # It is performed in Database.set_timings() which sets timings relative
        # to the SHUT stream in order to be synchronized with the video stream.
        # Alternatively, self.set_timings(*self.fit_timestamps()) does the job.

    @property
    def description(self) -> str:
        """Description of the stream, from the STNM record, if present."""
        return cast(str, self.metadata.get("STNM", ""))

    # Units may be understood by humans but cannot be parsed automatically
    # e.g. deg, deg, m, m/s, m/s, -, s, -, - is recorded as degdegmm/sm/ss.

    @property
    def units(self) -> str:
        """Units of the values, from the SIUN or UNIT record, if present."""
        if "SIUN" in self.metadata:
            return cast(str, self.metadata["SIUN"])
        if "UNIT" in self.metadata:
            return cast(str, self.metadata["UNIT"])
        return ""

    def fit_timestamps(self) -> tuple[float, float]:
        # The total number of samples includes the number of samples in the
        # current packet, which is known at the time the packet is written,
        # while the timestamp is for the first sample in the packet and the
        # timestamp of the next sample isn't known yet.
        samples = cast(
            list[int],
            [record.get_child("TSMP").value() for record in self.records],
        )
        self.num_samples = samples.pop()
        if not samples:
            raise ValueError("stream is too short: only one packet")
        samples = [0] + samples
        timestamps = cast(
            list[float],
            [record.get_child("STMP").value() for record in self.records],
        )
        result = scipy.stats.linregress(samples, timestamps)
        if result.rvalue**2 < 0.999:
            raise ValueError(
                f"stream isn't sampled at a constant rate: rvalue = {result.rvalue}"
            )
        if result.pvalue > 0.001:
            raise ValueError(
                f"stream isn't sampled at a constant rate: pvalue = {result.pvalue}"
            )
        return (
            float(result.slope) / 1_000_000,  # sample duration in seconds
            float(result.intercept) / 1_000_000,  # start timestamp in seconds
        )

    def set_timings(self, sample_duration: float, start_timestamp: float) -> None:
        """Set timing information for the stream."""
        self.sample_duration = sample_duration
        self.start_timestamp = start_timestamp
        self.total_duration = self.num_samples * sample_duration

    def timestamps(self) -> Sequence[float]:
        """
        Return timestamps for each sample in the stream.

        Raises :exc:`ValueError` if the stream contains only one GPMF packet
        i.e. is shorter than one second, or isn't sampled at a constant rate.

        """
        return [
            index * self.sample_duration + self.start_timestamp
            for index in range(self.num_samples)
        ]

    def timestamps_as_array(self) -> np.typing.NDArray[np.number]:
        """
        Return timestamps for each sample in the stream as a NumPy array.

        Raises :exc:`ValueError` if the stream contains only one GPMF packet
        i.e. is shorter than one second, or isn't sampled at a constant rate.

        """
        return np.arange(self.num_samples) * self.sample_duration + self.start_timestamp

    def values(
        self,
    ) -> (
        Sequence[Data]
        | Sequence[tuple[Data, ...]]
        | Sequence[Sequence[Data]]
        | Sequence[Sequence[tuple[Data, ...]]]
    ):
        """
        Return values for each sample in the stream.

        Depending on the type of the stream, this is a list of values or a list
        of tuples. When each sample can contain several values, then it's a list
        of lists of values or a list of lists of tuples.

        """
        values: (
            Sequence[Data]
            | Sequence[tuple[Data, ...]]
            | Sequence[Sequence[Data]]
            | Sequence[Sequence[tuple[Data, ...]]]
        )
        all_children = [record.get_children(self.fourcc) for record in self.records]
        if all(len(children) == 1 for children in all_children):
            values = [
                value for children in all_children for value in children[0].values()
            ]  # type: ignore
            if "SCAL" in self.metadata:
                scale = self.metadata["SCAL"]
                if isinstance(scale, tuple):
                    values = [
                        tuple(v / s for v, s in zip(value, scale))  # type: ignore
                        for value in values
                    ]
                elif isinstance(values[0], tuple):
                    values = [
                        tuple(v / scale for v in value)  # type: ignore
                        for value in values
                    ]
                else:
                    values = [value / scale for value in values]
        else:
            values = [child.values() for children in all_children for child in children]  # type: ignore
            if "SCAL" in self.metadata:
                scale = self.metadata["SCAL"]
                if isinstance(scale, tuple):
                    values = [
                        [tuple(v / s for v, s in zip(value, scale)) for value in sample]  # type: ignore
                        for sample in values
                    ]
                elif isinstance(
                    next(value for sample in values for value in sample),  # type: ignore
                    tuple,
                ):
                    values = [
                        [tuple(v / scale for v in value) for value in values]  # type: ignore
                        for sample in values
                    ]
                else:
                    values = [
                        [value / scale for value in sample]  # type: ignore
                        for sample in values
                    ]
        return values

    def values_as_array(self) -> np.typing.NDArray[np.number]:
        """
        Return values for each sample in the stream as a NumPy array.

        Depending on the type of the stream, this is a 1D-array, a 2D-array, or
        a 1D-array of structured data types.

        """
        all_children = [record.get_children(self.fourcc) for record in self.records]
        if not all(len(children) == 1 for children in all_children):
            raise ValueError("multiple values per sample")
        values: np.typing.NDArray[np.number] = np.concatenate(
            [children[0].values_as_array() for children in all_children]
        )
        if "SCAL" in self.metadata:
            scale = self.metadata["SCAL"]
            if isinstance(scale, tuple):
                fields = values.dtype.fields
                columns = [
                    values[f] / s  # type: ignore
                    for f, s in zip(fields, scale)  # type: ignore
                ]
                values = np.empty(
                    len(values),
                    dtype=np.dtype([("", column.dtype) for column in columns]),
                )
                for f, column in zip(fields, columns):  # type: ignore
                    values[f] = column  # type: ignore
            else:
                values = values / scale
        return values


class Database:
    def __init__(self, streams: dict[str, Stream], metadata: Metadata):
        self.streams = streams
        self.metadata = metadata

    @classmethod
    def from_mp4(cls, mp4_path: str | pathlib.Path) -> Self:
        """Create a database from an MP4 file."""
        mp4_file = MP4File(mp4_path)
        video_metadata: Metadata = {
            "width": mp4_file.video_metadata["width"],
            "height": mp4_file.video_metadata["height"],
            "frame_rate": mp4_file.video_metadata["frame_rate"],  # Fraction
            "duration": mp4_file.video_metadata["duration"],  # Decimal
            "nb_frames": mp4_file.video_metadata["nb_frames"],
        }

        raw_gpmf = mp4_file.extract_gpmf()
        gpmf = parse_gpmf(raw_gpmf)
        nb_packets = mp4_file.gpmd_metadata["nb_frames"]
        devc_metadata, strm_records = cls.extract_strm(gpmf, nb_packets)
        streams = cls.group_strm(strm_records)
        cls.set_timings(streams, video_metadata)

        return cls(streams, {**video_metadata, **devc_metadata})

    @classmethod
    def extract_strm(
        cls, records: Sequence[Record], nb_packets: int
    ) -> tuple[Metadata, Sequence[Record]]:
        """Check tree structure of GPMF data and extract STRM records."""
        if len(records) != nb_packets:
            raise ValueError(f"expected {nb_packets} packets, got {len(records)}")

        dvid_records: set[Record] = set()
        dvnm_records: set[Record] = set()
        strm_records: list[Record] = []
        for devc in records:
            if devc.fourcc != "DEVC":
                raise ValueError(
                    f"expected only DEVC records at the top level, got {devc.fourcc}"
                )
            for child in devc.children:
                match child.fourcc:
                    case "DVID":
                        dvid_records.add(child)
                    case "DVNM":
                        dvnm_records.add(child)
                    case "STRM":
                        strm_records.append(child)
                    case _:
                        raise ValueError(
                            f"expected only DVID, DVNM, and STRM records "
                            f"in DEVC records, got {child.fourcc}"
                        )

        if len(dvid_records) != 1 or len(dvnm_records) != 1:
            raise NotImplementedError("data from multiple devices")

        devc_metadata = {
            "device_id": dvid_records.pop().value(),
            "device_name": dvnm_records.pop().value(),
        }

        return devc_metadata, strm_records

    # These FourCC aren't constant metadata.
    VARIABLE_KEYS = {
        "TSMP",  # number of samples from start of capture to end of record
        "STMP",  # timestamp of first sample in record in microseconds
        "TMPC",  # device temperature in degrees Celsius
    }

    # These FourCC are ignored for the puproses of grouping.
    IGNORED_KEYS = {
        "TIMO",  # time offset of data in seconds; documented as rare
        "EMPT",  # number of payloads containing no new data
    }

    @classmethod
    def group_strm(cls, records: Sequence[Record]) -> dict[str, Stream]:
        """Group STRM records belonging to the same data stream."""

        def group_key(record: Record) -> str:
            return "".join(
                sorted(
                    set(child.fourcc for child in record.children) - cls.IGNORED_KEYS
                )
            )

        streams = {}
        for _, group_iter in itertools.groupby(
            sorted(records, key=group_key),
            key=group_key,
        ):
            group: Sequence[Record] = list(group_iter)
            metadata, group = cls.extract_metadata(group)

            keys = (
                set(child.fourcc for stream in group for child in stream.children)
                - cls.IGNORED_KEYS
                - cls.VARIABLE_KEYS
            )
            if len(keys) != 1:
                raise ValueError(f"failed to identify unique FourCC for stream: {keys}")
            fourcc = keys.pop()

            streams[fourcc] = Stream(fourcc, group, metadata)

        return streams

    @classmethod
    def extract_metadata(
        cls,
        records: Sequence[Record],
    ) -> tuple[Metadata, Sequence[Record]]:
        """Extract constant metadata children from a list of records."""
        if not records:
            return {}, []

        shared_children = set(records[0].children)
        for record in records[1:]:
            shared_children &= set(record.children)

        metadata = {}
        for fourcc, group_iter in itertools.groupby(
            shared_children,
            key=lambda record: record.fourcc,
        ):
            if fourcc in cls.VARIABLE_KEYS:
                continue
            group = list(group_iter)
            if len(group) != 1:
                continue
            record = group[0]
            if record.type == "":
                continue
            values = record.values()
            if len(values) != 1:
                continue
            value = values[0]
            metadata[fourcc] = value

        records = [
            dataclasses.replace(
                record,
                children=[
                    child for child in record.children if child.fourcc not in metadata
                ],
            )
            for record in records
        ]
        return metadata, records

    @classmethod
    def set_timings(
        cls, streams: dict[str, Stream], video_metadata: dict[str, Any]
    ) -> None:
        """Set timing information for each stream."""
        for stream in streams.values():
            if stream.fourcc == "LOGS":
                continue
            sample_duration, start_timestamp = stream.fit_timestamps()
            # Tolerate data streams finishing within +/- 0.5 second
            if not math.isclose(
                stream.num_samples * sample_duration,
                video_metadata["duration"],
                abs_tol=0.5,
            ):
                raise ValueError(
                    f"expected a {stream.fourcc} stream "
                    f"lasting {video_metadata['duration']} seconds, "
                    f"got {stream.num_samples * sample_duration} seconds"
                )
            stream.set_timings(sample_duration, start_timestamp)

        # Adjust timings relative to the SHUT stream. GoPro's demo does this,
        # presumably because SHUT is synchronized with the video stream.
        shut_start_timestamp = streams["SHUT"].start_timestamp
        for stream in streams.values():
            if stream.fourcc == "LOGS":
                continue
            stream.start_timestamp -= shut_start_timestamp


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} video.mp4\n")
        sys.exit(2)

    db = Database.from_mp4(sys.argv[1])

    print("Device ID:", db.metadata["device_id"])
    print("Device name:", db.metadata["device_name"])
    print()
    print("Streams:")
    for fourcc, stream in sorted(db.streams.items()):
        values = stream.values()
        description = ""
        if stream.description:
            description += f"{stream.description} "
        description += f"[{len(values)} records]"
        if hasattr(stream, "total_duration"):
            description += f" [{stream.total_duration:.3f} seconds]"
        print()
        print(f"{fourcc}: {description}")
        print()
        if len(values) < 10:
            for index in range(len(values)):
                print(f"{index:>4} ", values[index])
        else:
            print(" ...")
            for index in range(len(values) // 10, len(values) // 10 + 3):
                print(f"{index:>4} ", values[index])
            print(" ...")
            for index in range(9 * len(values) // 10 - 3, 9 * len(values) // 10):
                print(f"{index:>4} ", values[index])
            print(" ...")
