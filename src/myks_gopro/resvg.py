import enum
import pathlib
import sys
from collections.abc import Buffer
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_size_t,
    c_uint32,
    c_void_p,
    cast,
    create_string_buffer,
    pointer,
)
from typing import Any, Self, overload


__all__ = [
    "Error",
    "ResvgError",
    "ImageRendering",
    "ShapeRendering",
    "TextRendering",
    "Transform",
    "Size",
    "Rect",
    "transform_identity",
    "init_log",
    "Options",
    "RenderTree",
]


class CTypesEnum(enum.IntEnum):
    @classmethod
    def from_param(cls, obj: Any) -> int:
        return int(obj)


class Error(CTypesEnum):
    """List of possible errors."""

    OK = 0
    ERROR_NOT_AN_UTF8_STR = 1
    ERROR_FILE_OPEN_FAILED = 2
    ERROR_MALFORMED_GZIP = 3
    ERROR_ELEMENTS_LIMIT_REACHED = 4
    ERROR_INVALID_SIZE = 5
    ERROR_PARSING_FAILED = 6


class ResvgError(Exception):
    message_map = {
        Error.ERROR_NOT_AN_UTF8_STR: "Only UTF-8 content are supported.",
        Error.ERROR_FILE_OPEN_FAILED: "Failed to open the provided file.",
        Error.ERROR_MALFORMED_GZIP: "Compressed SVG must use the GZip algorithm.",
        Error.ERROR_ELEMENTS_LIMIT_REACHED: "SVG has more than 1_000_000 elements.",
        Error.ERROR_INVALID_SIZE: "SVG doesn't have a valid size.",
        Error.ERROR_PARSING_FAILED: "Failed to parse an SVG data.",
    }

    def __init__(self, error: Error):
        assert error != Error.OK, "resvg returned OK"
        self.error = error
        super().__init__(self.message_map[error])


class ImageRendering(CTypesEnum):
    """A image rendering method."""

    OPTIMIZE_QUALITY = 0
    OPTIMIZE_SPEED = 1


class ShapeRendering(CTypesEnum):
    """A shape rendering method."""

    OPTIMIZE_SPEED = 0
    CRISP_EDGES = 1
    GEOMETRIC_PRECISION = 2


class TextRendering(CTypesEnum):
    """A text rendering method."""

    OPTIMIZE_SPEED = 0
    OPTIMIZE_LEGIBILITY = 1
    GEOMETRIC_PRECISION = 2


class _Options(Structure):
    pass


class _RenderTree(Structure):
    pass


class Transform(Structure):
    """A 2D transform representation."""

    _fields_ = [
        ("a", c_float),
        ("b", c_float),
        ("c", c_float),
        ("d", c_float),
        ("e", c_float),
        ("f", c_float),
    ]


class Size(Structure):
    """A size representation."""

    _fields_ = [
        ("width", c_float),
        ("height", c_float),
    ]


class Rect(Structure):
    """A rectangle representation."""

    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
    ]


resvg = CDLL("libresvg.dylib")

resvg.resvg_transform_identity.argtypes = []
resvg.resvg_transform_identity.restype = Transform


def transform_identity() -> Transform:
    """
    Creates an identity transform.

    """
    result = resvg.resvg_transform_identity()
    assert isinstance(result, Transform)
    return result


resvg.resvg_init_log.argtypes = []
resvg.resvg_init_log.restype = None


def init_log() -> None:
    """
    Initializes the library log.

    Use it if you want to see any warnings.

    Must be called only once.

    All warnings will be printed to the `stderr`.

    """
    resvg.resvg_init_log()


resvg.resvg_options_create.argtypes = []
resvg.resvg_options_create.restype = POINTER(_Options)

resvg.resvg_options_set_resources_dir.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_resources_dir.restype = None

resvg.resvg_options_set_dpi.argtypes = [
    POINTER(_Options),
    c_float,
]
resvg.resvg_options_set_dpi.restype = None

resvg.resvg_options_set_stylesheet.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_stylesheet.restype = None

resvg.resvg_options_set_font_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_font_family.restype = None

resvg.resvg_options_set_font_size.argtypes = [
    POINTER(_Options),
    c_float,
]
resvg.resvg_options_set_font_size.restype = None

resvg.resvg_options_set_serif_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_serif_family.restype = None

resvg.resvg_options_set_sans_serif_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_sans_serif_family.restype = None

resvg.resvg_options_set_cursive_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_cursive_family.restype = None

resvg.resvg_options_set_fantasy_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_fantasy_family.restype = None

resvg.resvg_options_set_monospace_family.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_monospace_family.restype = None

resvg.resvg_options_set_languages.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_set_languages.restype = None

resvg.resvg_options_set_shape_rendering_mode.argtypes = [
    POINTER(_Options),
    ShapeRendering,  # type: ignore
]
resvg.resvg_options_set_shape_rendering_mode.restype = None

resvg.resvg_options_set_text_rendering_mode.argtypes = [
    POINTER(_Options),
    TextRendering,  # type: ignore
]
resvg.resvg_options_set_text_rendering_mode.restype = None

resvg.resvg_options_set_image_rendering_mode.argtypes = [
    POINTER(_Options),
    ImageRendering,  # type: ignore
]
resvg.resvg_options_set_image_rendering_mode.restype = None

resvg.resvg_options_load_font_data.argtypes = [
    POINTER(_Options),
    c_char_p,
    c_size_t,
]
resvg.resvg_options_load_font_data.restype = None

resvg.resvg_options_load_font_file.argtypes = [
    POINTER(_Options),
    c_char_p,
]
resvg.resvg_options_load_font_file.restype = Error

resvg.resvg_options_load_system_fonts.argtypes = [POINTER(_Options)]
resvg.resvg_options_load_system_fonts.restype = None

resvg.resvg_options_destroy.argtypes = [POINTER(_Options)]
resvg.resvg_options_destroy.restype = None


class Options:
    def __init__(self, **kwargs: Any) -> None:
        """
        Create an Options object.

        Options may be set with keyword arguments or by assigning to attributes.

        """
        self._options = resvg.resvg_options_create()

        self._resources_dir: pathlib.Path | None = None
        self._dpi: float = 96
        self._stylesheet: str | None = None
        self._font_family = "Times New Roman"
        self._font_size: float = 12
        self._serif_family = "Times New Roman"
        self._sans_serif_family = "Arial"
        self._cursive_family = "Comic Sans MS"
        self._fantasy_family = "Papyrus" if sys.platform == "darwin" else "Impact"
        self._monospace_family = "Courier New"
        self._languages: list[str] | None = None
        self._shape_rendering_mode = ShapeRendering.GEOMETRIC_PRECISION
        self._text_rendering_mode = TextRendering.OPTIMIZE_LEGIBILITY
        self._image_rendering_mode = ImageRendering.OPTIMIZE_QUALITY

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def resources_dir(self) -> pathlib.Path | None:
        """
        Sets a directory that will be used during relative paths resolving.

        Expected to be the same as the directory that contains the SVG file,
        but can be set to any.

        """
        return self._resources_dir

    @resources_dir.setter
    def resources_dir(self, path: pathlib.Path | str) -> None:
        path = pathlib.Path(path)
        resvg.resvg_options_set_resources_dir(self._options, bytes(path))
        self._resources_dir = path

    @property
    def dpi(self) -> float:
        """
        Sets the target DPI.

        Impact units conversion.

        """
        return self._dpi

    @dpi.setter
    def dpi(self, dpi: float) -> None:
        resvg.resvg_options_set_dpi(self._options, float(dpi))
        self._dpi = dpi

    @property
    def stylesheet(self) -> str | None:
        """
        Content of a stylesheet that will be used when resolving CSS attributes.

        """
        return self._stylesheet

    @stylesheet.setter
    def stylesheet(self, content: str | None) -> None:
        resvg.resvg_options_set_stylesheet(
            self._options,
            None if content is None else content.encode(),
        )
        self._stylesheet = content

    @property
    def font_family(self) -> str:
        """
        Sets the default font family.

        Will be used when no ``font-family`` attribute is set in the SVG.

        """
        return self._font_family

    @font_family.setter
    def font_family(self, family: str) -> None:
        resvg.resvg_options_set_font_family(self._options, family.encode())
        self._font_family = family

    @property
    def font_size(self) -> float:
        """
        Sets the default font size.

        Will be used when no ``font-size`` attribute is set in the SVG.

        """
        return self._font_size

    @font_size.setter
    def font_size(self, size: float) -> None:
        resvg.resvg_options_set_font_size(self._options, float(size))
        self._font_size = size

    @property
    def serif_family(self) -> str:
        """
        Sets the ``serif`` font family.

        Has no effect when the ``text`` feature is not enabled.

        """
        return self._serif_family

    @serif_family.setter
    def serif_family(self, family: str) -> None:
        resvg.resvg_options_set_serif_family(self._options, family.encode())
        self._serif_family = family

    @property
    def sans_serif_family(self) -> str:
        """
        Sets the ``sans-serif`` font family.

        Has no effect when the ``text`` feature is not enabled.

        """
        return self._sans_serif_family

    @sans_serif_family.setter
    def sans_serif_family(self, family: str) -> None:
        resvg.resvg_options_set_sans_serif_family(self._options, family.encode())
        self._sans_serif_family = family

    @property
    def cursive_family(self) -> str:
        """
        Sets the ``cursive`` font family.

        Has no effect when the ``text`` feature is not enabled.

        """
        return self._cursive_family

    @cursive_family.setter
    def cursive_family(self, family: str) -> None:
        resvg.resvg_options_set_cursive_family(self._options, family.encode())
        self._cursive_family = family

    @property
    def fantasy_family(self) -> str:
        """
        Sets the ``fantasy`` font family.

        Has no effect when the ``text`` feature is not enabled.

        """
        return self._fantasy_family

    @fantasy_family.setter
    def fantasy_family(self, family: str) -> None:
        resvg.resvg_options_set_fantasy_family(self._options, family.encode())
        self._fantasy_family = family

    @property
    def monospace_family(self) -> str:
        """
        Sets the ``monospace`` font family.

        Has no effect when the ``text`` feature is not enabled.

        """
        return self._monospace_family

    @monospace_family.setter
    def monospace_family(self, family: str) -> None:
        resvg.resvg_options_set_monospace_family(self._options, family.encode())
        self._monospace_family = family

    @property
    def languages(self) -> list[str] | None:
        """
        Sets a comma-separated list of languages.

        Will be used to resolve a ``systemLanguage`` conditional attribute.

        Example: ``["en", "en-US"]``.

        """
        return self._languages

    @languages.setter
    def languages(self, languages: list[str] | None) -> None:
        resvg.resvg_options_set_languages(
            self._options,
            None if languages is None else ",".join(languages).encode(),
        )
        self._languages = languages

    @property
    def shape_rendering_mode(self) -> ShapeRendering:
        """
        Sets the default shape rendering method.

        Will be used when an SVG element's ``shape-rendering`` property is ``auto``.

        """
        return self._shape_rendering_mode

    @shape_rendering_mode.setter
    def shape_rendering_mode(self, mode: ShapeRendering) -> None:
        resvg.resvg_options_set_shape_rendering_mode(self._options, mode)
        self._shape_rendering_mode = mode

    @property
    def text_rendering_mode(self) -> TextRendering:
        """
        Sets the default text rendering method.

        Will be used when an SVG element's ``text-rendering`` property is ``auto``.

        """
        return self._text_rendering_mode

    @text_rendering_mode.setter
    def text_rendering_mode(self, mode: TextRendering) -> None:
        resvg.resvg_options_set_text_rendering_mode(self._options, mode)
        self._text_rendering_mode = mode

    @property
    def image_rendering_mode(self) -> ImageRendering:
        """
        Sets the default image rendering method.

        Will be used when an SVG element's ``image-rendering`` property is ``auto``.

        """
        return self._image_rendering_mode

    @image_rendering_mode.setter
    def image_rendering_mode(self, mode: ImageRendering) -> None:
        resvg.resvg_options_set_image_rendering_mode(self._options, mode)
        self._image_rendering_mode = mode

    def load_font_data(self, data: bytes) -> None:
        """
        Loads a font data into the internal fonts database.

        Prints a warning into the log when the data is not a valid TrueType font.

        Has no effect when the ``text`` feature is not enabled.

        """
        resvg.resvg_options_load_font_data(self._options, data, len(data))

    def load_font_file(self, file_path: pathlib.Path | str) -> None:
        """
        Loads a font file into the internal fonts database.

        Prints a warning into the log when the data is not a valid TrueType font.

        Has no effect when the ``text`` feature is not enabled.

        """
        file_path = pathlib.Path(file_path)
        result = resvg.resvg_options_load_font_file(self._options, bytes(file_path))
        if result != Error.OK:
            raise ResvgError(result.name)

    def load_system_fonts(self) -> None:
        """
        Loads system fonts into the internal fonts database.

        This method is very IO intensive.

        This method should be executed only once.

        The system scanning is not perfect, so some fonts may be omitted.
        Please send a bug report in this case.

        Prints warnings into the log.

        Has no effect when the ``text`` feature is not enabled.

        """
        resvg.resvg_options_load_system_fonts(self._options)

    def __del__(self) -> None:
        resvg.resvg_options_destroy(self._options)


resvg.resvg_parse_tree_from_file.argtypes = [
    c_char_p,
    POINTER(_Options),
    POINTER(POINTER(_RenderTree)),
]
resvg.resvg_parse_tree_from_file.restype = Error

resvg.resvg_parse_tree_from_data.argtypes = [
    c_char_p,
    c_void_p,
    POINTER(_Options),
    POINTER(POINTER(_RenderTree)),
]
resvg.resvg_parse_tree_from_data.restype = Error

resvg.resvg_is_image_empty.argtypes = [POINTER(_RenderTree)]
resvg.resvg_is_image_empty.restype = c_bool

resvg.resvg_get_image_size.argtypes = [POINTER(_RenderTree)]
resvg.resvg_get_image_size.restype = Size

resvg.resvg_get_object_bbox.argtypes = [
    POINTER(_RenderTree),
    POINTER(Rect),
]
resvg.resvg_get_object_bbox.restype = c_bool

resvg.resvg_get_image_bbox.argtypes = [
    POINTER(_RenderTree),
    POINTER(Rect),
]
resvg.resvg_get_image_bbox.restype = c_bool

resvg.resvg_node_exists.argtypes = [
    POINTER(_RenderTree),
    c_char_p,
]
resvg.resvg_node_exists.restype = c_bool

resvg.resvg_get_node_transform.argtypes = [
    POINTER(_RenderTree),
    c_char_p,
    POINTER(Transform),
]
resvg.resvg_get_node_transform.restype = c_bool

resvg.resvg_get_node_bbox.argtypes = [
    POINTER(_RenderTree),
    c_char_p,
    POINTER(Rect),
]
resvg.resvg_get_node_bbox.restype = c_bool

resvg.resvg_get_node_stroke_bbox.argtypes = [
    POINTER(_RenderTree),
    c_char_p,
    POINTER(Rect),
]
resvg.resvg_get_node_stroke_bbox.restype = c_bool

resvg.resvg_tree_destroy.argtypes = [POINTER(_RenderTree)]
resvg.resvg_tree_destroy.restype = None

resvg.resvg_render.argtypes = [
    POINTER(_RenderTree),
    Transform,
    c_uint32,
    c_uint32,
    POINTER(c_char),
]
resvg.resvg_render.restype = None

resvg.resvg_render_node.argtypes = [
    POINTER(_RenderTree),
    c_char_p,
    Transform,
    c_uint32,
    c_uint32,
    POINTER(c_char),
]
resvg.resvg_render_node.restype = c_bool


class RenderTree:
    def __init__(self) -> None:
        self._tree = cast(c_void_p(), POINTER(_RenderTree))

    @classmethod
    def from_file(
        cls,
        file_path: pathlib.Path | str,
        options: Options | None = None,
    ) -> Self:
        """
        Creates a render tree from file.

        .svg and .svgz files are supported.

        """
        file_path = pathlib.Path(file_path)
        if options is None:
            options = Options()

        tree = cls()
        result = resvg.resvg_parse_tree_from_file(
            bytes(file_path),
            options._options,
            pointer(tree._tree),
        )
        if result != Error.OK:
            raise ResvgError(Error(result))
        return tree

    @classmethod
    def from_data(cls, data: str | bytes, options: Options | None = None) -> Self:
        """
        Creates a render tree from data.

        ``data`` can be a SVG string or gzip compressed data.

        """
        if isinstance(data, str):
            data = data.encode()
        if options is None:
            options = Options()

        tree = cls()
        result = resvg.resvg_parse_tree_from_data(
            data,
            len(data),
            options._options,
            pointer(tree._tree),
        )
        if result != Error.OK:
            raise ResvgError(Error(result))
        return tree

    def is_image_empty(self) -> bool:
        """
        Return ``True`` if the tree doesn't have any nodes.

        """
        result = resvg.resvg_is_image_empty(self._tree)
        assert isinstance(result, bool)
        return result

    def get_image_size(self) -> Size:
        """
        Return the image size.

        The size of an image that is required to render this SVG.

        Note that elements outside the viewbox will be clipped. This is by design.
        If you want to render the whole SVG content, use ``get_image_bbox`` instead.

        """
        result = resvg.resvg_get_image_size(self._tree)
        assert isinstance(result, Size)
        return result

    def get_object_bbox(self) -> Rect | None:
        """
        Return the object bounding box.

        This bounding box does not include objects stroke and filter regions.
        This is what SVG calls "absolute object bounding box".
        If you're looking for a "complete" bounding box see ``get_image_bbox``.

        """
        bbox = Rect()
        if not resvg.resvg_get_object_bbox(self._tree, byref(bbox)):
            return None
        return bbox

    def get_image_bbox(self) -> Rect | None:
        """
        Returns the image bounding box.

        This bounding box contains the maximum SVG dimensions.
        It can be bigger or smaller than ``get_image_size``.
        Use it when you want to avoid clipping of elements that are outside
        the SVG viewbox.

        """
        bbox = Rect()
        if not resvg.resvg_get_image_bbox(self._tree, byref(bbox)):
            return None
        return bbox

    def node_exists(self, node_id: str) -> bool:
        """
        Returns ``True`` if a renderable node with this ID exists.

        """
        result = resvg.resvg_node_exists(self._tree, node_id.encode())
        assert isinstance(result, bool)
        return result

    def get_node_transform(self, node_id: str) -> Transform | None:
        """
        Returns the node's transform.

        """
        transform = Transform()
        if not resvg.resvg_get_node_transform(
            self._tree,
            node_id.encode(),
            byref(transform),
        ):
            return None
        return transform

    def get_node_bbox(self, node_id: str) -> Rect | None:
        """
        Returns the node's bounding box in canvas coordinates.

        """
        bbox = Rect()
        if resvg.resvg_get_node_bbox(
            self._tree,
            node_id.encode(),
            byref(bbox),
        ):
            return bbox
        return None

    def get_node_stroke_bbox(self, node_id: str) -> Rect | None:
        """
        Returns the node's bounding box, including stroke, in canvas coordinates.

        """
        bbox = Rect()
        if not resvg.resvg_get_node_stroke_bbox(
            self._tree,
            node_id.encode(),
            byref(bbox),
        ):
            return None
        return bbox

    @overload
    def render(
        self,
        pixmap: None,
        width: int | None = None,
        height: int | None = None,
        transform: Transform | None = None,
    ) -> bytes: ...

    @overload
    def render(
        self,
        pixmap: Buffer,
        width: int,
        height: int,
        transform: Transform | None = None,
    ) -> None: ...

    def render(
        self,
        pixmap: Buffer | None = None,
        width: int | None = None,
        height: int | None = None,
        transform: Transform | None = None,
    ) -> bytes | None:
        """
        Render the render tree onto the pixmap.

        If pixmap is provided, render the image onto the buffer and return None.

        Else, render the image onto a new buffer and return it as bytes.

        """
        if transform is None:
            transform = transform_identity()
        if pixmap is None:
            if width is None or height is None:
                size = self.get_image_size()
                if width is None:
                    width = int(size.width)
                if height is None:
                    height = int(size.height)
            _pixmap = create_string_buffer(width * height * 4)
            resvg.resvg_render(
                self._tree,
                transform,
                width,
                height,
                _pixmap,
            )
            return _pixmap.raw
        else:
            if width is None or height is None:
                raise ValueError(
                    "width and height must be provided when pixmap is provided"
                )
            resvg.resvg_render(
                self._tree,
                transform,
                width,
                height,
                (c_char * (width * height * 4)).from_buffer(pixmap),
            )
            return None

    def render_node(
        self,
        node_id: str,
        pixmap: Buffer,
        width: int,
        height: int,
        transform: Transform | None = None,
    ) -> bool:
        """
        Render the node onto the pixmap.

        """
        if transform is None:
            transform = transform_identity()
        result = resvg.resvg_render_node(
            self._tree,
            node_id.encode(),
            transform,
            width,
            height,
            (c_char * (width * height * 4)).from_buffer(pixmap),
        )
        assert isinstance(result, bool)
        return result

    def __del__(self) -> None:
        """Destroys the render tree."""
        if self._tree:
            resvg.resvg_tree_destroy(self._tree)
