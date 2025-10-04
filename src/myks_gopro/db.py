"""
Extract and expose GPMF data.

"""

from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys
from typing import Any, Self, Sequence, cast

import numpy as np

from .gpmf import Data, Record, parse_gpmf
from .mp4 import MP4File


__all__ = ["Database"]


Metadata = dict[str, Any]


class Stream:
    def __init__(self, fourcc: str, records: Sequence[Record], metadata: Metadata):
        self.fourcc = fourcc
        self.records = records
        self.metadata = metadata

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
        mp4_file.check_gpmf_packets_sequence()
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

    @classmethod
    def group_strm(cls, records: Sequence[Record]) -> dict[str, Stream]:
        """Group STRM records belonging to the same data stream."""

        def group_key(record: Record) -> str:
            return "".join(sorted(set(child.fourcc for child in record.children)))

        streams = {}
        for _, group_iter in itertools.groupby(
            sorted(records, key=group_key),
            key=group_key,
        ):
            group: Sequence[Record] = list(group_iter)
            metadata, group = cls.extract_metadata(group)

            keys = set(child.fourcc for stream in group for child in stream.children)
            keys -= cls.VARIABLE_KEYS
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
        description = f"{stream.description} " if stream.description else ""
        values = stream.values()
        total_samples = len(values)
        print()
        print(f"{fourcc}: {description}[{total_samples} records]")
        print()
        print(" ...")
        for index in range(total_samples // 10, total_samples // 10 + 3):
            print(f"{index:>4} ", values[index])
        print(" ...")
        for index in range(9 * total_samples // 10, 9 * total_samples // 10 + 3):
            print(f"{index:>4} ", values[index])
        print(" ...")
