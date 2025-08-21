"""
Parse GoPro's GPMF format.

Extract all data from a GPMF stream in a convenient structure.

https://github.com/gopro/gpmf-parser/blob/main/docs/README.md

"""

from __future__ import annotations

import dataclasses
import datetime
import struct
from typing import Self, Sequence, cast
import uuid

import numpy as np


__all__ = ["parse_gpmf", "Record"]


Data = str | int | float | datetime.datetime | uuid.UUID

Metadata = dict[str, Data | tuple[Data, ...]]


class ChildNotFound(LookupError):
    pass


class MultipleChildren(LookupError):
    pass


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

NP_TYPES = {
    "b": np.dtype("b"),
    "B": np.dtype("B"),
    "d": np.dtype(">f8"),
    "f": np.dtype(">f4"),
    "j": np.dtype(">i8"),
    "J": np.dtype(">u8"),
    "l": np.dtype(">i4"),
    "L": np.dtype(">u4"),
    "q": np.dtype(">i4"),
    "Q": np.dtype(">i8"),
    "s": np.dtype(">i2"),
    "S": np.dtype(">u2"),
}


def convert_string(value: bytes) -> str:
    return value.rstrip(b"\x00").decode("iso-8859-1", errors="surrogateescape")


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
        fourcc = fourcc.decode("ascii")
        type_ = "" if type_ == b"\x00" else type_.decode("ascii")
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

    def parse_payload_as_array(self) -> np.typing.NDArray[np.number]:
        try:
            dtype = cast(np.dtype[np.number], NP_TYPES[self.type])
        except KeyError:
            raise ValueError(f"unsupported type {self.type}")

        array = np.frombuffer(self.raw_data, dtype=dtype)

        if self.type == "q":
            array = array / 65536.0
        elif self.type == "Q":
            array = array / 4294967296.0

        if self.size != dtype.itemsize:
            array = array.reshape(self.repeat, self.size // dtype.itemsize)

        return array

    def parse_complex_payload_as_array(self) -> np.typing.NDArray[np.number]:
        if self.struct_type == "":
            raise ValueError(f"complex type {self.fourcc} requires a TYPE")

        try:
            dtype = cast(
                np.dtype[np.number],
                np.dtype([("", TYPES[type_]) for type_ in self.struct_type]),
            )
        except KeyError:
            raise ValueError(f"unsupported type {self.type}")
        dtype = dtype.newbyteorder(">")

        array = np.frombuffer(self.raw_data, dtype=dtype)

        types = set(self.struct_type)
        if types == {"q"}:
            array = array / 65536.0
        elif types == {"Q"}:
            array = array / 4294967296.0
        elif types & {"q", "Q"}:
            raise NotImplementedError("complex type with Q notation")

        # Records may be tuples or structs but shouldn't be both.
        if self.size != dtype.itemsize:
            raise NotImplementedError("complex type with repetition")

        return array

    def array(self) -> np.typing.NDArray[np.number]:
        """
        Return the values from the record as a Numpy array.

        If the record is a simple type, return a list of values when repeat is
        equal to 1, else a list of tuples.

        If the record is a complex type, always return return a list of tuples.

        """
        match self.type:
            case "":
                raise ValueError(f"{self.fourcc} is a nested record")
            case "?":
                return self.parse_complex_payload_as_array()
            case _:
                return self.parse_payload_as_array()

    def get_child(self, fourcc: str) -> Self:
        children = self.get_children(fourcc)
        if not children:
            raise ChildNotFound(f"child {fourcc} not found in {self.fourcc}")
        if len(children) > 1:
            raise MultipleChildren(f"multiple {fourcc} children in {self.fourcc}")
        return children[0]

    def get_children(self, fourcc: str) -> Sequence[Self]:
        return [child for child in self.children if child.fourcc == fourcc]


def parse_gpmf(raw_gpmf: bytes, offset: int = 0) -> Sequence[Record]:
    """Parse a GPMF stream into a list of records."""
    records = []
    while offset < len(raw_gpmf):
        record, offset = Record.parse(raw_gpmf, offset)
        records.append(record)
    assert offset == len(raw_gpmf), "read beyond end of GPMF"
    return records
