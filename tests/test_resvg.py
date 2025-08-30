import gzip
import math
import pathlib
import tempfile
import unittest

from myks_gopro.resvg import (
    Error,
    ImageRendering,
    Options,
    RenderTree,
    ResvgError,
    ShapeRendering,
    TextRendering,
    transform_identity,
)


class TestTransform(unittest.TestCase):
    def test_transform_identity(self):
        transform = transform_identity()
        self.assertEqual(transform.a, 1)
        self.assertEqual(transform.b, 0)
        self.assertEqual(transform.c, 0)
        self.assertEqual(transform.d, 1)
        self.assertEqual(transform.e, 0)
        self.assertEqual(transform.f, 0)


class TestOptions(unittest.TestCase):
    def test_init(self):
        options = Options()
        self.assertEqual(options.dpi, 96)

    def test_init_with_options(self):
        options = Options(dpi=300, font_family="Helvetica")
        self.assertEqual(options.dpi, 300)
        self.assertEqual(options.font_family, "Helvetica")

    def test_set_resources_dir(self):
        options = Options(resources_dir="resources")
        self.assertEqual(options.resources_dir, pathlib.Path("resources"))

    def test_set_dpi(self):
        options = Options(dpi=72)
        self.assertEqual(options.dpi, 72)

    def test_set_stylesheet(self):
        options = Options(stylesheet="body { color: green; }")
        self.assertEqual(options.stylesheet, "body { color: green; }")

    def test_set_font_family(self):
        options = Options(font_family="Arial")
        self.assertEqual(options.font_family, "Arial")

    def test_set_font_size(self):
        options = Options(font_size=16)
        self.assertEqual(options.font_size, 16)

    def test_set_serif_family(self):
        options = Options(serif_family="Georgia")
        self.assertEqual(options.serif_family, "Georgia")

    def test_set_sans_serif_family(self):
        options = Options(sans_serif_family="Helvetica")
        self.assertEqual(options.sans_serif_family, "Helvetica")

    def test_set_cursive_family(self):
        options = Options(cursive_family="Brush Script MT")
        self.assertEqual(options.cursive_family, "Brush Script MT")

    def test_set_fantasy_family(self):
        options = Options(fantasy_family="Chalkduster")
        self.assertEqual(options.fantasy_family, "Chalkduster")

    def test_set_monospace_family(self):
        options = Options(monospace_family="Monaco")
        self.assertEqual(options.monospace_family, "Monaco")

    def test_set_languages(self):
        options = Options(languages=["fr", "fr-FR"])
        self.assertEqual(options.languages, ["fr", "fr-FR"])

    def test_set_shape_rendering_mode(self):
        options = Options(shape_rendering_mode=ShapeRendering.CRISP_EDGES)
        self.assertEqual(options.shape_rendering_mode, ShapeRendering.CRISP_EDGES)

    def test_set_text_rendering_mode(self):
        options = Options(text_rendering_mode=TextRendering.OPTIMIZE_LEGIBILITY)
        self.assertEqual(options.text_rendering_mode, TextRendering.OPTIMIZE_LEGIBILITY)

    def test_set_image_rendering_mode(self):
        options = Options(image_rendering_mode=ImageRendering.OPTIMIZE_SPEED)
        self.assertEqual(options.image_rendering_mode, ImageRendering.OPTIMIZE_SPEED)


SVG = """\
<?xml version="1.0" encoding="utf-8"?>
<svg width="128" height="128" viewBox="-8 -8 16 16"
    version="1.1" xmlns="http://www.w3.org/2000/svg">
  <circle id="circle" cx="0" cy="0" r="6" fill="green"
    stroke="darkgreen" stroke-width="1" />
  <rect x="-4" y="-1" width="8" height="2" fill="white" />
  <rect x="-1" y="-4" width="2" height="8" fill="white" />
</svg>
"""

SVGZ = gzip.compress(SVG.encode())


class TestRenderTree(unittest.TestCase):
    def test_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / "test.svg"
            temp_file.write_text(SVG)
            render_tree = RenderTree.from_file(temp_file)
        self.assertFalse(render_tree.is_image_empty())

    def test_from_compressed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / "test.svgz"
            temp_file.write_bytes(SVGZ)
            render_tree = RenderTree.from_file(temp_file)
        self.assertFalse(render_tree.is_image_empty())

    def test_from_file_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / "test.svg"
            with self.assertRaises(ResvgError) as raised:
                RenderTree.from_file(temp_file)
        self.assertEqual(raised.exception.error, Error.ERROR_FILE_OPEN_FAILED)

    def test_from_data(self):
        render_tree = RenderTree.from_data(SVG)
        self.assertFalse(render_tree.is_image_empty())

    def test_from_compressed_data(self):
        render_tree = RenderTree.from_data(SVGZ)
        self.assertFalse(render_tree.is_image_empty())

    def test_from_data_error(self):
        with self.assertRaises(ResvgError) as raised:
            RenderTree.from_data(b"this is not SVG")
        self.assertEqual(raised.exception.error, Error.ERROR_PARSING_FAILED)

    def test_is_image_empty(self):
        render_tree = RenderTree.from_data("<svg></svg>")
        self.assertTrue(render_tree.is_image_empty())

    def test_get_image_size(self):
        render_tree = RenderTree.from_data(SVG)
        size = render_tree.get_image_size()
        self.assertEqual(size.width, 128)
        self.assertEqual(size.height, 128)

    def test_get_object_bbox(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_object_bbox()
        self.assertEqual(bbox.x, 16)
        self.assertEqual(bbox.y, 16)
        self.assertEqual(bbox.width, 96)
        self.assertEqual(bbox.height, 96)

    def test_get_object_bbox_none(self):
        render_tree = RenderTree.from_data("<svg></svg>")
        self.assertIsNone(render_tree.get_object_bbox())

    def test_get_image_bbox(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_image_bbox()
        self.assertTrue(math.isclose(bbox.x, 12, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.y, 12, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.width, 104, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.height, 104, rel_tol=0.001))

    def test_get_image_bbox_none(self):
        render_tree = RenderTree.from_data("<svg></svg>")
        self.assertIsNone(render_tree.get_image_bbox())

    def test_node_exists(self):
        render_tree = RenderTree.from_data(SVG)
        self.assertTrue(render_tree.node_exists("circle"))
        self.assertFalse(render_tree.node_exists("square"))

    def test_get_node_transform(self):
        render_tree = RenderTree.from_data(SVG)
        transform = render_tree.get_node_transform("circle")
        self.assertEqual(transform.a, 8)
        self.assertEqual(transform.b, 0)
        self.assertEqual(transform.c, 0)
        self.assertEqual(transform.d, 8)
        self.assertEqual(transform.e, 64)
        self.assertEqual(transform.f, 64)

    def test_get_node_transform_none(self):
        render_tree = RenderTree.from_data(SVG)
        transform = render_tree.get_node_transform("square")
        self.assertIsNone(transform)

    def test_get_node_bbox(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_node_bbox("circle")
        self.assertEqual(bbox.x, 16)
        self.assertEqual(bbox.y, 16)
        self.assertEqual(bbox.width, 96)
        self.assertEqual(bbox.height, 96)

    def test_get_node_bbox_none(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_node_bbox("square")
        self.assertIsNone(bbox)

    def test_get_node_stroke_bbox(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_node_stroke_bbox("circle")
        self.assertTrue(math.isclose(bbox.x, 12, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.y, 12, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.width, 104, rel_tol=0.001))
        self.assertTrue(math.isclose(bbox.height, 104, rel_tol=0.001))

    def test_get_node_stroke_bbox_none(self):
        render_tree = RenderTree.from_data(SVG)
        bbox = render_tree.get_node_stroke_bbox("square")
        self.assertIsNone(bbox)

    @staticmethod
    def get_pixel(pixmap, x, y, width):
        offset = (y * width + x) * 4
        return tuple(pixmap[offset : offset + 4])

    def test_render(self):
        render_tree = RenderTree.from_data(SVG)
        pixmap = bytearray(128 * 128 * 4)
        self.assertIsNone(render_tree.render(pixmap, 128, 128))
        self.assertEqual(self.get_pixel(pixmap, 32, 32, 128), (0, 100, 0, 255))
        self.assertEqual(self.get_pixel(pixmap, 64, 64, 128), (255, 255, 255, 255))

    def test_render_new(self):
        render_tree = RenderTree.from_data(SVG)
        pixmap = render_tree.render()
        self.assertEqual(self.get_pixel(pixmap, 48, 48, 128), (0, 128, 0, 255))
        self.assertEqual(self.get_pixel(pixmap, 64, 64, 128), (255, 255, 255, 255))

    def test_render_node(self):
        render_tree = RenderTree.from_data(SVG)
        pixmap = bytearray(128 * 128 * 4)
        # Translate the node to its intended position. It isn't entirely clear
        # how render_node is expected to work there's a scaled viewBox.
        bbox = render_tree.get_node_bbox("circle")
        transform = render_tree.get_node_transform("circle")
        assert transform.b == 0
        assert transform.c == 0
        transform.e += bbox.x * transform.a
        transform.f += bbox.y * transform.d
        self.assertTrue(render_tree.render_node("circle", pixmap, 128, 128, transform))
        self.assertEqual(self.get_pixel(pixmap, 48, 48, 128), (0, 128, 0, 255))
        self.assertEqual(self.get_pixel(pixmap, 64, 64, 128), (0, 128, 0, 255))
        self.assertFalse(render_tree.render_node("square", pixmap, 128, 128))
