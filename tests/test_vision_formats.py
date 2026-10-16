"""A picture reaches a vision model only in a format the servers decode, and a picture that is not attached is never
described as attached.

Both found by the host application model benchmark (2026-10-16):

- An AVIF went to every vision model as ``data:application/octet-stream`` (image_mime knows no AVIF magic), and
  LM Studio answered HTTP 400 "'url' field must be a base64 encoded image." -- the analysis failed instead of
  describing the file from its metadata.
- A picture sent without its pixels (a profile without vision, a picture too large or cut short) got the prompt "The
  image itself is attached. Describe what it shows", the invitation build_image_metadata_prompt exists to avoid; text
  models answered with a circle, a star or a shield on a picture that is one blue area.
"""
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from sift.config import Config
from sift.extract.image import VISION_FORMATS, _dimensions, _mime
from sift.llm.base import LLMResult, image_mime
from sift.summarize import summarize_path
from tests import _fixtures as fx

REPLY = "TYPE: image\nSUMMARY: A picture.\nKEY POINTS:\n- one\nENTITIES: none\nNOTES: none\n"
#: The start of an AVIF file: a box size, "ftyp", the brand.
AVIF = struct.pack(">I", 28) + b"ftypavif" + b"\x00\x00\x00\x00" + b"avifmif1miaf" + b"\x00" * 64
TIFF = b"II*\x00" + struct.pack("<I", 8) + b"\x00" * 64
ATTACHED = "The image itself is attached"
CANNOT_SEE = "you CANNOT see this image"


class Recorder:
    provider = "dummy"
    model = "dummy-model"
    endpoint = "memory://dummy"

    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def list_models(self):
        return [self.model]

    def generate(self, system, user, *, images=None, temperature=0.2, max_tokens=800):
        self.calls.append({"user": user, "images": images})
        return LLMResult(text=REPLY, model=self.model, provider=self.provider)


class VisionFormatTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def summarize(self, name, data, **config):
        path = self.dir / name
        path.write_bytes(data)
        client = Recorder()
        summ = summarize_path(path, Config(**config), client=client)
        self.assertEqual(len(client.calls), 1, client.calls)
        return summ, client.calls[0]

    # happy path
    def test_a_png_goes_to_the_vision_model_as_attached(self):
        summ, call = self.summarize("a.png", fx.make_png(self.dir / "src.png").read_bytes())
        self.assertEqual(len(call["images"]), 1)
        self.assertIn(ATTACHED, call["user"])
        self.assertTrue(summ.llm_used)

    # sad path
    def test_an_avif_is_not_sent_and_the_model_is_told_it_cannot_see_it(self):
        summ, call = self.summarize("a.avif", AVIF)
        self.assertFalse(call["images"])
        self.assertIn(CANNOT_SEE, call["user"])
        self.assertNotIn(ATTACHED, call["user"])
        self.assertTrue(summ.llm_used)
        self.assertTrue(any("AVIF" in n and "not shown" in n for n in summ.notes), summ.notes)

    def test_a_tiff_is_not_sent(self):
        _summ, call = self.summarize("a.tif", TIFF)
        self.assertFalse(call["images"])
        self.assertIn(CANNOT_SEE, call["user"])

    def test_webp_and_bmp_are_not_sent(self):
        """Both went out before and both got HTTP 400 from LM Studio with their true type (GGUF and MLX alike)."""
        webp = b"RIFF" + struct.pack("<I", 30) + b"WEBPVP8 " + b"\x00" * 10 + struct.pack("<HH", 4, 4)
        bmp = (b"BM" + (1000).to_bytes(4, "little") + b"\x00" * 4 + (54).to_bytes(4, "little")
               + (40).to_bytes(4, "little") + struct.pack("<ii", 4, 4) + b"\x00" * 8)
        for name, data in (("a.webp", webp), ("a.bmp", bmp)):
            _summ, call = self.summarize(name, data)
            self.assertFalse(call["images"], name)
            self.assertIn(CANNOT_SEE, call["user"], name)

    def test_a_picture_without_vision_is_not_described_as_attached(self):
        """vision off: the pixels stay home, and the prompt says so."""
        _summ, call = self.summarize("a.png", fx.make_png(self.dir / "src.png").read_bytes(), vision=False)
        self.assertFalse(call["images"])
        self.assertIn(CANNOT_SEE, call["user"])
        self.assertNotIn(ATTACHED, call["user"])

    # edge cases
    def test_avif_is_named_by_its_brand(self):
        self.assertEqual(_dimensions(AVIF)[0], "AVIF")
        self.assertEqual(_mime("AVIF"), "image/avif")

    def test_a_picture_too_large_for_the_prompt_is_not_described_as_attached(self):
        from sift.extract import image
        data = fx.make_png(self.dir / "src.png").read_bytes()
        cap = image.VISION_BYTE_CAP
        image.VISION_BYTE_CAP = len(data) - 1
        self.addCleanup(setattr, image, "VISION_BYTE_CAP", cap)
        _summ, call = self.summarize("a.png", data)
        self.assertFalse(call["images"])
        self.assertIn(CANNOT_SEE, call["user"])

    def test_the_summarizer_holds_the_line_when_the_extractor_lets_one_through(self):
        """Second layer: the extractor's gate widened by mistake still sends no octet-stream."""
        from unittest import mock
        with mock.patch("sift.extract.image.VISION_FORMATS", VISION_FORMATS | {"AVIF"}):
            summ, call = self.summarize("a.avif", AVIF)
        self.assertFalse(call["images"])
        self.assertIn(CANNOT_SEE, call["user"])

    def test_every_format_sent_is_one_image_mime_names_truly(self):
        """The data: URI states the type image_mime reads off the bytes: a format let through must be one it knows,
        or the server gets application/octet-stream again."""
        self.assertTrue(VISION_FORMATS)
        for fmt in VISION_FORMATS:
            self.assertNotEqual(_mime(fmt), "image/*", fmt)
        samples = {"PNG": fx.make_png(self.dir / "p.png").read_bytes(), "JPEG": b"\xff\xd8\xff\xe0" + b"\x00" * 16,
                   "GIF": b"GIF89a" + b"\x00" * 16}
        for fmt in VISION_FORMATS:
            self.assertEqual(image_mime(samples[fmt]), _mime(fmt), fmt)


if __name__ == "__main__":
    unittest.main()
