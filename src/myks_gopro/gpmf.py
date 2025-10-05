"""
Parse GoPro's GPMF format.

Extract all data from a GPMF stream in a convenient structure.

https://github.com/gopro/gpmf-parser/blob/main/docs/README.md

"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import struct
import sys
import uuid
from typing import Self, Sequence, TextIO, cast

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
    """KLV record."""

    # Key - 7-bit ASCII
    fourcc: str

    # Data type - a single letter for simple types (see TYPES or NP_TYPES), "?"
    # for complex type (structure defined by a previous TYPE record), or "" for
    # nested records. (\x00 is converted to "" during parsing for convenience.)
    type: str

    # Structure size - each sample is limited to 255 bytes or less.
    size: int

    # Repeat - number of samples in the record, up to 65535.
    repeat: int

    # Value - payload of the record, which may be data or nested records.
    raw_data: memoryview

    # Children - nested records - used only when type is "".
    children: Sequence[Self]

    # Structure of a complex record - required when type is "?".
    struct_type: str

    def __hash__(self) -> int:
        return hash((self.fourcc, self.type, self.size, self.repeat, self.raw_data))

    def dump(
        self,
        show_values: bool = True,
        indent: int = 0,
        file: TextIO = sys.stdout,
    ) -> None:
        """Dump the record to a file."""
        print(
            " " * indent
            + f"{self.fourcc} type={self.type} size={self.size} repeat={self.repeat}",
            file=file,
        )
        indent += 4
        if self.type == "":
            for child in self.children:
                child.dump(show_values=show_values, indent=indent, file=file)
        elif show_values:
            for value in self.values():
                print(" " * indent + repr(value), file=file)

    @classmethod
    def parse(
        cls,
        raw_gpmf: bytes,
        offset: int = 0,
        struct_type: str = "",
    ) -> tuple[Self, int]:
        """
        Parse a record from a GPMF stream, starting at the given offset.

        Payloads are parsed lazily, when calling :meth:`value`, :meth:`values`,
        or :meth:`values_as_array`.

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
                    value = child.value()
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

    def get_children(self, fourcc: str) -> Sequence[Self]:
        """Get a list of children with the given FourCC."""
        return [child for child in self.children if child.fourcc == fourcc]

    def get_child(self, fourcc: str) -> Self:
        """Get a single child with the given FourCC."""
        children = self.get_children(fourcc)
        if not children:
            raise ChildNotFound(f"child {fourcc} not found in {self.fourcc}")
        if len(children) > 1:
            raise MultipleChildren(f"multiple {fourcc} children in {self.fourcc}")
        return children[0]

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

    def value(self) -> Data | tuple[Data, ...]:
        """
        Return a single value from the record.

        Raises :exc:`ValueError` if the record contains more than one value.
        """
        values = self.values()
        if len(values) != 1:
            raise ValueError(f"expected 1 {self.fourcc} value, found {len(values)}")
        return values[0]

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

    def values_as_array(self) -> np.typing.NDArray[np.number]:
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


def parse_gpmf(raw_gpmf: bytes, offset: int = 0) -> Sequence[Record]:
    """Parse a GPMF stream into a list of records."""
    records = []
    while offset < len(raw_gpmf):
        record, offset = Record.parse(raw_gpmf, offset)
        records.append(record)
    assert offset == len(raw_gpmf), "read beyond end of GPMF"
    return records


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} video.gpmf video.txt\n")
        sys.exit(2)

    gpmf_file = pathlib.Path(sys.argv[1])
    txt_file = pathlib.Path(sys.argv[2])
    gpmf = parse_gpmf(gpmf_file.read_bytes())
    with txt_file.open("w") as handle:
        for record in gpmf:
            record.dump(file=handle)
