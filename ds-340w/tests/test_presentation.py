"""Check the proposal's explicit artifact and template contracts."""

import json
from pathlib import Path
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class PresentationTests(unittest.TestCase):
    def test_method_and_evaluation_slides_omit_removed_captions(self):
        removed = {
            4: [
                "Implemented: renderer, checks, and pair export. Training is proposed.",
                "Use language-model DPO, not a diffusion denoising loss.",
                "Token-level DPO",
            ],
            5: ["Count invalid outputs. Keep test references out of training."],
            6: ["Next: label pairs → train adapter → test"],
        }
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        with ZipFile(ROOT / "presentation/training-ai-to-pixelate.pptx") as package:
            for number, captions in removed.items():
                slide = ET.fromstring(package.read(f"ppt/slides/slide{number}.xml"))
                labels = [node.text or "" for node in slide.findall(".//a:t", ns)]
                for caption in captions:
                    with self.subTest(slide=number, caption=caption):
                        self.assertNotIn(caption, labels)
                if number == 4:
                    self.assertIn("DPO", labels)

    def test_parent_papers_have_read_aloud_summaries(self):
        content = json.loads((ROOT / "presentation/content.json").read_text())
        papers = content["slides"][1]["papers"]
        with ZipFile(ROOT / "presentation/training-ai-to-pixelate.pptx") as package:
            slide = ET.fromstring(package.read("ppt/slides/slide2.xml"))
        ns = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        shapes = {
            "".join(node.text or "" for node in shape.findall(".//a:t", ns)): shape
            for shape in slide.findall(".//p:sp", ns)
        }
        for paper in papers:
            with self.subTest(paper=paper["name"]):
                self.assertTrue(paper["idea"].endswith("."))
                self.assertGreaterEqual(len(paper["idea"].split()), 12)
                self.assertLessEqual(len(paper["idea"].split()), 25)
                summary = shapes[paper["idea"]].find(".//a:xfrm", ns)
                connection = shapes[paper["transfer"]].find(".//a:xfrm", ns)
                summary_bottom = int(summary.find("a:off", ns).get("y")) + int(
                    summary.find("a:ext", ns).get("cy"))
                self.assertLessEqual(summary_bottom, int(connection.find("a:off", ns).get("y")))

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
