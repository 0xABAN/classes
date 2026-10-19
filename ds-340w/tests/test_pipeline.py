"""Behavioral checks for the pilot, including invalid model outputs."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path("pixelcode/renderer.py").exists(), "renderer not implemented")
        self.renderer = importlib.import_module("pixelcode.renderer")

    def test_rectangles_render_on_exact_grid(self):
        image = self.renderer.render('rect(2, 3, 2, 1, "#ff0000");')
        self.assertEqual(image.size, (32, 32))
        self.assertEqual(image.getpixel((2, 3)), (255, 0, 0))
        self.assertEqual(image.getpixel((3, 3)), (255, 0, 0))
        self.assertEqual(image.getpixel((4, 3)), (245, 245, 244))

    def test_arbitrary_javascript_and_out_of_bounds_are_rejected(self):
        for code in ["fetch('https://example.com');", "while(true){}", "",
                     'rect(31,0,2,1,"#ffffff");', 'rect(0,0,0,1,"#ffffff");',
                     'rect(0.5,0,1,1,"#ffffff");', 'rect(-1,0,1,1,"#ffffff");']:
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.renderer.render(code)

    def test_rejects_trailing_garbage_and_excessive_commands(self):
        code = 'rect(0,0,1,1,"#000000");'
        for invalid in [code + "evil()", code * 513]:
            with self.assertRaises(ValueError):
                self.renderer.render(invalid)

    def test_dataset_is_deterministic_and_has_disjoint_scenes(self):
        from pixelcode.dataset import build_records
        rows = build_records()
        self.assertEqual(rows, build_records())
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(r["split"] == "train" for r in rows), 100)
        self.assertEqual(sum(r["split"] == "test" for r in rows), 20)
        self.assertEqual(len({r["scene_id"] for r in rows}), 120)
        self.assertEqual(len({r["prompt"] for r in rows}), 120)
        for row in rows:
            self.assertEqual(self.renderer.render(row["reference_code"]).size, (32, 32))

    def test_reference_programs_pass_structural_checks(self):
        from pixelcode.dataset import build_records
        from pixelcode.evaluation import check_candidate
        for row in build_records():
            result = check_candidate(row, row["reference_code"])
            self.assertTrue(result["valid"], row["id"])
            self.assertTrue(result["palette_ok"], row["id"])
            self.assertTrue(result["placement_ok"], row["id"])

    def test_invalid_output_is_a_failure_not_dropped(self):
        from pixelcode.dataset import build_records
        from pixelcode.evaluation import evaluate
        rows = build_records()[:2]
        outputs = {rows[0]["id"]: rows[0]["reference_code"], rows[1]["id"]: "bad();"}
        result = evaluate(rows, outputs)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["valid_rate"], 0.5)
        self.assertEqual(result["palette_rate"], 0.5)

    def test_missing_and_unknown_predictions_are_errors(self):
        from pixelcode.dataset import build_records
        from pixelcode.evaluation import evaluate
        rows = build_records()[:1]
        for predictions in [{}, {"unknown": "bad();"}]:
            with self.assertRaises(ValueError):
                evaluate(rows, predictions)

    def test_recolor_preserves_geometry_and_unrelated_pixels(self):
        from pixelcode.dataset import build_records
        from pixelcode.evaluation import check_recolor
        row = build_records()[0]
        before = row["reference_code"]
        after = before.replace(row["color_hex"], "#b8579e")
        self.assertTrue(check_recolor(before, after, row["color_hex"], "#b8579e"))
        self.assertFalse(check_recolor(before, before, row["color_hex"], "#b8579e"))

    def test_preference_export_requires_human_labels_and_train_ids(self):
        from pixelcode.dataset import build_records
        from pixelcode.preferences import training_pair
        rows = build_records()
        train = next(r for r in rows if r["split"] == "train")
        test = next(r for r in rows if r["split"] == "test")
        a = train["reference_code"]
        b = a.replace(train["color_hex"], "#b8579e")
        pair = training_pair(train, a, b, "a", "reviewer-1")
        self.assertEqual(pair["chosen"], a)
        self.assertEqual(pair["rejected"], b)
        for row, label, reviewer in [(test, "a", "reviewer-1"), (train, "tie", "reviewer-1"),
                                     (train, "a", "")]:
            with self.assertRaises(ValueError):
                training_pair(row, a, b, label, reviewer)

    def test_demo_cli_creates_reproducible_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-m", "pixelcode", "demo", "--output", folder],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(folder)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["source"], "hand_authored_reference_programs")
            self.assertEqual(report["n"], 120)
            self.assertEqual(report["valid_rate"], 1.0)
            self.assertTrue(report["recolor_check"])
            self.assertTrue((output / "mushroom.png").is_file())

    def test_dataset_cli_and_evaluation_cli(self):
        from pixelcode.dataset import build_records
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            dataset = output / "prompts.jsonl"
            result = subprocess.run([sys.executable, "-m", "pixelcode", "dataset", "--output", str(dataset)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in dataset.read_text().splitlines()]
            self.assertEqual(rows, build_records())
            candidates = output / "candidates.jsonl"
            candidates.write_text("\n".join(json.dumps({"id": r["id"], "code": r["reference_code"]})
                                            for r in rows if r["split"] == "test"))
            result = subprocess.run([sys.executable, "-m", "pixelcode", "evaluate", "--dataset", str(dataset),
                                     "--predictions", str(candidates), "--split", "test"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["n"], 20)


if __name__ == "__main__":
    unittest.main()
