"""Check the proposal's explicit artifact and template contracts."""

import json
from pathlib import Path
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class PresentationTests(unittest.TestCase):
    def test_six_slides_and_ten_minute_split(self):
        path = ROOT / "presentation/training-ai-to-pixelate.pptx"
        self.assertTrue(path.exists(), "presentation is not built")
        with ZipFile(path) as package:
            slides = [n for n in package.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            self.assertEqual(len(slides), 6)
            body = " ".join(" ".join(ET.fromstring(package.read(n)).itertext()) for n in slides)
            for term in ["AutomaTikZ", "Text2Reward", "Diffusion-DPO", "120 prompts", "100 train / 20 test"]:
                self.assertIn(term, body)
            for obsolete in ["High Bias", "High Variance", "PH0", "TODO", "PLACEHOLDER"]:
                self.assertNotIn(obsolete, body)
            for i in range(1, 7):
                self.assertIn(f"ppt/notesSlides/notesSlide{i}.xml", package.namelist())
            dimensions = ET.fromstring(package.read("ppt/presentation.xml")).find(
                "{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz")
            self.assertEqual((dimensions.attrib["cx"], dimensions.attrib["cy"]), ("12191999", "6858000"))
        content = json.loads((ROOT / "presentation/content.json").read_text())
        self.assertEqual(sum(content["timing_seconds"][:3]), 300)
        self.assertEqual(sum(content["timing_seconds"][3:]), 300)


if __name__ == "__main__":
    unittest.main()
