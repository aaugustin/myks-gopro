"""
Parse GoPro's GPMF format.

Extract all data from a GPMF stream in a convenient structure.

https://github.com/gopro/gpmf-parser/blob/main/docs/README.md

"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import pathlib
import struct
import sys
from typing import Self, Sequence
import uuid


__all__ = ["parse_gpmf", "extract_streams"]


Data = str | int | float | datetime.datetime | uuid.UUID

Metadata = dict[str, Data | tuple[Data, ...]]


TYPES = {
    "b": "b",
    "B": "B",
    # See special case in parse_payload for "c".
    "d": "d",
    "f": "f",
    "F": "4s",
    "G": "8s",
    "j": "q",
    "J": "Q",
    "l": "i",
    "L": "I",
    "q": "i",
    "Q": "q",
    "s": "h",
    "S": "H",
    "U": "20s",
}


def convert_string(value: bytes) -> str:
    return value.rstrip(b"\x00").decode(errors="surrogateescape")


def convert_uuid(value: bytes) -> uuid.UUID:
    return uuid.UUID(bytes=value)


def convert_q15_16(value: int) -> float:
    return value / 32768.0


def convert_q31_32(value: int) -> float:
    return value / 2147483648.0


def convert_gopro_datetime(value: str) -> datetime.datetime:
    dt = datetime.datetime.strptime(value, "%y%m%d%H%M%S.%f")
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def keep(value: Data) -> Data:
    return value


CONVERTERS = {
    "c": convert_string,
    "F": convert_string,
    "G": convert_uuid,
    "q": convert_q15_16,
    "Q": convert_q31_32,
    "U": convert_gopro_datetime,
}

# These FourCC aren't constant metadata.
VARIABLE_KEYS = {"TSMP", "STMP", "TMPC"}


@dataclasses.dataclass
class Record:
    fourcc: str
    type: str  # \x00 is converted to the empty string.
    size: int
    repeat: int
    raw_data: memoryview

    children: Sequence[Self]
    struct_type: str

    def __hash__(self) -> int:
        return hash((self.fourcc, self.type, self.size, self.repeat, self.raw_data))

    @classmethod
    def parse(
        cls,
        raw_gpmf: bytes,
        offset: int = 0,
        struct_type: str = "",
    ) -> tuple[Self, int]:
        """
        Parse a record from a GPMF stream, starting at the given offset.

        Payloads are parsed lazily, when calling :meth:`values()`.

        """
        fourcc, type_, size, repeat = struct.unpack_from(">4scBH", raw_gpmf, offset)
        fourcc = fourcc.decode()
        type_ = "" if type_ == b"\x00" else type_.decode()
        offset += 8

        raw_data = memoryview(raw_gpmf)[offset : offset + size * repeat]
        if type_ == "":
            sub_offset = offset
            offset += size * repeat
            children = []
            while sub_offset < offset:
                child, sub_offset = cls.parse(raw_gpmf, sub_offset, struct_type)
                children.append(child)
                if child.fourcc == "TYPE":
                    value = child.values()[0]
                    assert isinstance(value, str)
                    struct_type = value
            assert sub_offset == offset, "read beyond end of record"
        else:
            # Data is 4-bytes aligned. Round up to the next multiple of 4.
            offset += ((size * repeat + 3) // 4) * 4
            children = []

        if type_ == "?":
            if struct_type == "":
                raise ValueError(f"complex type {fourcc} without TYPE")
        else:
            struct_type = ""

        # Scale and units should be tuples, not lists, but GoPro got it wrong.
        if fourcc in {"SCAL", "SIUN", "UNIT"} and repeat > 1:
            size, repeat = size * repeat, 1

        return cls(fourcc, type_, size, repeat, raw_data, children, struct_type), offset

    def parse_payload(self) -> Sequence[Data] | Sequence[tuple[Data, ...]]:
        # We need the size of variable length fields to build the format string.
        fmt = f"{self.size}s" if self.type == "c" else TYPES[self.type]
        size = struct.calcsize(fmt)
        num = self.size // size
        fmt *= num
        fmt = f">{fmt}"
        assert struct.calcsize(fmt) == self.size, "size mismatch"

        entries: Sequence[tuple[Data, ...]]
        entries = list(struct.iter_unpack(fmt, self.raw_data))

        converter = CONVERTERS.get(self.type)
        if converter is not None:
            entries = [
                tuple(converter(value) for value in entry)  # type: ignore
                for entry in entries
            ]

        if num == 1:
            return [entry[0] for entry in entries]
        else:
            return entries

    def parse_complex_payload(self) -> Sequence[tuple[Data, ...]]:
        if self.struct_type == "":
            raise ValueError(f"complex type {self.fourcc} requires a TYPE")
        if "c" in self.struct_type:
            raise NotImplementedError(
                f"complex type {self.fourcc} with variable length field"
            )

        fmt = "".join(TYPES[type_] for type_ in self.struct_type)
        size = struct.calcsize(fmt)
        num = self.size // size
        fmt *= num
        fmt = f">{fmt}"
        assert struct.calcsize(fmt) == self.size, "size mismatch"

        entries = list(struct.iter_unpack(fmt, self.raw_data))

        converters = [CONVERTERS.get(type_, keep) for type_ in self.struct_type * num]
        if any(converter is not keep for converter in converters):
            entries = [
                tuple(converter(value) for converter, value in zip(converters, entry))  # type: ignore
                for entry in entries
            ]

        return entries

    def values(self) -> Sequence[Data] | Sequence[tuple[Data, ...]]:
        """
        Return the values from the record.

        If the record is a simple type, return a list of values when repeat is
        equal to 1, else a list of tuples.

        If the record is a complex type, always return return a list of tuples.

        """
        match self.type:
            case "":
                raise ValueError(f"{self.fourcc} is a nested record")
            case "?":
                return self.parse_complex_payload()
            case _:
                return self.parse_payload()


def parse_gpmf(raw_gpmf: bytes, offset: int = 0) -> Sequence[Record]:
    """Parse a GPMF stream into a list of records."""
    records = []
    while offset < len(raw_gpmf):
        record, offset = Record.parse(raw_gpmf, offset)
        records.append(record)
    assert offset == len(raw_gpmf), "read beyond end of GPMF"
    return records


def extract_metadata(records: Sequence[Record]) -> tuple[Metadata, Sequence[Record]]:
    """Extract shared metadata from a list of records."""
    shared_children = set.intersection(*(set(record.children) for record in records))

    metadata = {}
    for fourcc, group_iter in itertools.groupby(
        shared_children,
        key=lambda record: record.fourcc,
    ):
        if fourcc in VARIABLE_KEYS:
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
            children=[ch for ch in record.children if ch.fourcc not in metadata],
        )
        for record in records
    ]
    return metadata, records


def extract_streams(
    devices: Sequence[Record],
    verbose: bool = False,
) -> tuple[Metadata, dict[str, tuple[Metadata, Sequence[Record]]]]:
    """"""
    if not all(device.fourcc == "DEVC" for device in devices):
        raise ValueError("GPMF stream must contain DEVC records at the top level")

    metadata, devices = extract_metadata(devices)

    if not all(ch.fourcc == "STRM" for device in devices for ch in device.children):
        raise NotImplementedError("GPMF stream contains records from multiple devices")

    streams = [stream for device in devices for stream in device.children]

    def group_key(record: Record) -> str:
        return "".join(sorted(set(child.fourcc for child in record.children)))

    streams_by_fourcc = {}
    for _, group_iter in itertools.groupby(
        sorted(streams, key=group_key),
        key=group_key,
    ):
        group: Sequence[Record] = list(group_iter)
        stream_metadata, group = extract_metadata(group)
        keys = set(child.fourcc for stream in group for child in stream.children)
        keys -= VARIABLE_KEYS
        if len(keys) != 1:
            raise ValueError(f"failed to identify unique FourCC for stream: {keys}")
        streams_by_fourcc[keys.pop()] = stream_metadata, group

    return metadata, streams_by_fourcc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} video.gpmf\n")
        sys.exit(2)

    gpmf = pathlib.Path(sys.argv[1]).read_bytes()
    records = parse_gpmf(gpmf)
    track_metadata, streams = extract_streams(records)

    print("Device ID:", track_metadata.get("DVID", "-"))
    print("Device name:", track_metadata.get("DVNM", "-"))
    print("Streams:")
    for fourcc, (stream_metadata, stream_data) in sorted(streams.items()):
        description = stream_metadata.get("STNM", "-")
        num_records = sum(record.repeat for record in stream_data)
        print(f"* {fourcc}: {description} ({num_records} records)")
