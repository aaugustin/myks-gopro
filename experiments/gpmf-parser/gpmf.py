import collections
import ctypes
import enum
import functools
import pathlib
import sys


# These definitions must be synchronized with GPMF_common.h and GPMF_parser.h.

class CTypesEnum(enum.IntEnum):
    @classmethod
    def from_param(cls, obj):
        return int(obj)

class GPMFError(CTypesEnum):
    OK = 0
    ERROR_MEMORY = 1
    ERROR_BAD_STRUCTURE = 2
    ERROR_BUFFER_END = 3
    ERROR_FIND = 4
    ERROR_LAST = 5
    ERROR_TYPE_NOT_SUPPORTED = 6
    ERROR_SCALE_NOT_SUPPORTED = 7
    ERROR_SCALE_COUNT = 8
    ERROR_UNKNOWN_TYPE = 9
    ERROR_RESERVED = 10

class GPMFSampleType(CTypesEnum):
    STRING_ASCII = ord(b"c")
    SIGNED_BYTE = ord(b"b")
    UNSIGNED_BYTE = ord(b"B")
    SIGNED_SHORT = ord(b"s")
    UNSIGNED_SHORT = ord(b"S")
    FLOAT = ord(b"f")
    FOURCC = ord(b"F")
    SIGNED_LONG = ord(b"l")
    UNSIGNED_LONG = ord(b"L")
    Q15_16_FIXED_POINT = ord(b"q")
    Q31_32_FIXED_POINT = ord(b"Q")
    SIGNED_64BIT_INT = ord(b"j")
    UNSIGNED_64BIT_INT = ord(b"J")
    DOUBLE = ord(b"d")
    STRING_UTF8 = ord(b"u")
    UTC_DATE_TIME = ord(b"U")
    GUID = ord(b"G")

    COMPLEX = ord(b"?")
    COMPRESSED = ord(b"#")

    NEST = 0

    EMPTY = 0xFE
    ERROR = 0xFF

class GPMFLevels(CTypesEnum):
    CURRENT_LEVEL = 0
    RECURSE_LEVELS = 1
    TOLERANT = 2

GPMF_NEST_LIMIT = 16

class GPMFStream(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.POINTER(ctypes.c_uint32)),
        ("buffer_size_longs", ctypes.c_uint32),
        ("pos", ctypes.c_uint32),
        ("last_level_pos", ctypes.c_uint32 * GPMF_NEST_LIMIT),
        ("nest_size", ctypes.c_uint32 * GPMF_NEST_LIMIT),
        ("last_seek", ctypes.c_uint32 * GPMF_NEST_LIMIT),
        ("nest_level", ctypes.c_uint32),
        ("device_count", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("device_name", ctypes.c_char * 32),
        ("cbhandle", ctypes.c_size_t),
    ]


# Map all functions from GPMF_parser.h.

gpmf_parser = ctypes.CDLL(pathlib.Path(__file__).with_name("GPMF_parser.so"))

gpmf_init = gpmf_parser.GPMF_Init
gpmf_init.argtypes = [ctypes.POINTER(GPMFStream), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
gpmf_init.restype = GPMFError

gpmf_reset_state = gpmf_parser.GPMF_ResetState
gpmf_reset_state.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_reset_state.restype = GPMFError

gpmf_copy_state = gpmf_parser.GPMF_CopyState
gpmf_copy_state.argtypes = [ctypes.POINTER(GPMFStream), ctypes.POINTER(GPMFStream)]
gpmf_copy_state.restype = GPMFError

gpmf_validate = gpmf_parser.GPMF_Validate
gpmf_validate.argtypes = [ctypes.POINTER(GPMFStream), GPMFLevels]
gpmf_validate.restype = GPMFError

gpmf_next = gpmf_parser.GPMF_Next
gpmf_next.argtypes = [ctypes.POINTER(GPMFStream), GPMFLevels]
gpmf_next.restype = GPMFError

gpmf_find_prev = gpmf_parser.GPMF_FindPrev
gpmf_find_prev.argtypes = [ctypes.POINTER(GPMFStream), ctypes.c_uint32, GPMFLevels]
gpmf_find_prev.restype = GPMFError

gpmf_find_next = gpmf_parser.GPMF_FindNext
gpmf_find_next.argtypes = [ctypes.POINTER(GPMFStream), ctypes.c_uint32, GPMFLevels]
gpmf_find_next.restype = GPMFError

gpmf_seek_to_samples = gpmf_parser.GPMF_SeekToSamples
gpmf_seek_to_samples.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_seek_to_samples.restype = GPMFError

gpmf_key = gpmf_parser.GPMF_Key
gpmf_key.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_key.restype = ctypes.c_uint32

gpmf_type = gpmf_parser.GPMF_Type
gpmf_type.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_type.restype = GPMFSampleType

gpmf_struct_size = gpmf_parser.GPMF_StructSize
gpmf_struct_size.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_struct_size.restype = ctypes.c_uint32

gpmf_repeat = gpmf_parser.GPMF_Repeat
gpmf_repeat.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_repeat.restype = ctypes.c_uint32

gpmf_payload_sample_count = gpmf_parser.GPMF_PayloadSampleCount
gpmf_payload_sample_count.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_payload_sample_count.restype = ctypes.c_uint32

gpmf_elements_in_struct = gpmf_parser.GPMF_ElementsInStruct
gpmf_elements_in_struct.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_elements_in_struct.restype = ctypes.c_uint32

gpmf_raw_data_size = gpmf_parser.GPMF_RawDataSize
gpmf_raw_data_size.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_raw_data_size.restype = ctypes.c_uint32

gpmf_raw_data = gpmf_parser.GPMF_RawData
gpmf_raw_data.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_raw_data.restype = ctypes.c_void_p

gpmf_size_of_type = gpmf_parser.GPMF_SizeofType
gpmf_size_of_type.argtypes = [GPMFSampleType]
gpmf_size_of_type.restype = ctypes.c_uint32

gpmf_expand_complex_type = gpmf_parser.GPMF_ExpandComplexTYPE
gpmf_expand_complex_type.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
gpmf_expand_complex_type.restype = ctypes.c_uint32

gpmf_size_of_complex_type = gpmf_parser.GPMF_SizeOfComplexTYPE
gpmf_size_of_complex_type.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
gpmf_size_of_complex_type.restype = ctypes.c_uint32

gpmf_reserved = gpmf_parser.GPMF_Reserved
gpmf_reserved.argtypes = [ctypes.c_uint32]
gpmf_reserved.restype = GPMFError

gpmf_formatted_data_size = gpmf_parser.GPMF_FormattedDataSize
gpmf_formatted_data_size.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_formatted_data_size.restype = ctypes.c_uint32

gpmf_scaled_data_size = gpmf_parser.GPMF_ScaledDataSize
gpmf_scaled_data_size.argtypes = [ctypes.POINTER(GPMFStream), GPMFSampleType]
gpmf_scaled_data_size.restype = ctypes.c_uint32

gpmf_formatted_data = gpmf_parser.GPMF_FormattedData
gpmf_formatted_data.argtypes = [ctypes.POINTER(GPMFStream), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
gpmf_formatted_data.restype = GPMFError

gpmf_scaled_data = gpmf_parser.GPMF_ScaledData
gpmf_scaled_data.argtypes = [ctypes.POINTER(GPMFStream), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, GPMFSampleType]
gpmf_scaled_data.restype = GPMFError

gpmf_free = gpmf_parser.GPMF_Free
gpmf_free.argtypes = [ctypes.POINTER(GPMFStream)]
gpmf_free.restype = GPMFError


def camel_case(s):
    return "".join(word.capitalize() for word in s.split("_"))


class GPMFException(Exception):

    def __init__(self, function, error):
        self.function = function
        self.error = error

    def __str__(self):
        return f"GPMF_{self.function} failed: {self.error.name}"


class GPMF:

    def __init__(self, raw_gpmf, validate=True):
        self.stream = GPMFStream()

        if len(raw_gpmf) % 4 != 0:
            raise ValueError("GPMF data must be 32-bits aligned")
        # Keep a reference to the buffer to prevent it from being garbage collected.
        self.buffer = (ctypes.c_uint32 * (len(raw_gpmf) // 4)).from_buffer_copy(raw_gpmf)
        self.init(self.buffer, len(raw_gpmf))

        if validate:
            self.validate(GPMFLevels.RECURSE_LEVELS)
            self.reset_state()

    def __getattr__(self, method_name):
        function = globals()[f"gpmf_{method_name}"]

        @functools.wraps(function)
        def function_with_error_check(*args, **kwargs):
            result = function(ctypes.byref(self.stream), *args, **kwargs)
            if isinstance(result, GPMFError):
                if result != GPMFError.OK:
                    raise GPMFException(camel_case(method_name), result)
            else:
                return result

        return function_with_error_check


def parse_gpmf(raw_gpmf):
    gpmf = GPMF(raw_gpmf)
    data = collections.defaultdict(list)
    while True:
        try:
            gpmf.next(GPMFLevels.RECURSE_LEVELS)
        except GPMFException as exc:
            if exc.error == GPMFError.ERROR_BUFFER_END:
                break
            else:
                raise

        key = gpmf.key()
        fourcc = key.to_bytes(4, "little").decode()
        type_ = gpmf.type()
        elements = gpmf.elements_in_struct()
        samples = gpmf.repeat()

        if type_ in (
            GPMFSampleType.FOURCC,
            GPMFSampleType.STRING_ASCII,
            GPMFSampleType.NEST,
        ):
            continue

        if samples:
            buffer = (ctypes.c_double * (elements * samples))()
            buffer_size = elements * samples * ctypes.sizeof(ctypes.c_double)
            gpmf.scaled_data(buffer, buffer_size, 0, samples, GPMFSampleType.DOUBLE)
            data[fourcc].extend(buffer)

    return data

if __name__ == "__main__":
    data = parse_gpmf(pathlib.Path(sys.argv[1]).read_bytes())
    for key, value in data.items():
        print(key, len(value), value[:10])
