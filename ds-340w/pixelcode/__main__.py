"""Run dataset generation, a local demo, or candidate evaluation."""

import argparse
import json
from pathlib import Path

from PIL import Image

from .dataset import build_records, read_records
from .evaluation import check_recolor, evaluate
from .renderer import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Write the deterministic prompt dataset")
    dataset.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="Render hand-authored examples and verify the pipeline")
    demo.add_argument("--output", type=Path, required=True)
    evaluator = commands.add_parser("evaluate", help="Evaluate a complete set of candidate programs")
    evaluator.add_argument("--dataset", type=Path, default=Path("data/prompts.jsonl"))
    evaluator.add_argument("--predictions", type=Path, required=True)
    evaluator.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()

    if args.command == "dataset":
        rows = build_records()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"Wrote {len(rows)} scenes: 100 train, 20 test")
        return

    if args.command == "evaluate":
        rows = [row for row in read_records(args.dataset) if row["split"] == args.split]
        predictions = read_records(args.predictions)
        print(json.dumps(evaluate(rows, {r["id"]: r["code"] for r in predictions}), indent=2))
        return

    rows = build_records()
    args.output.mkdir(parents=True, exist_ok=True)
    examples = [next(row for row in rows if row["object"] == name and row["split"] == "train")
                for name in ("mushroom", "tree", "house", "boat", "heart")]
    for row in examples:
        code = row["reference_code"]
        (args.output / f'{row["object"]}.js').write_text(code + "\n")
        render(code).resize((320, 320), Image.Resampling.NEAREST).save(args.output / f'{row["object"]}.png')

    example = examples[0]
    recolored = example["reference_code"].replace(example["color_hex"], "#b8579e")
    render(recolored).resize((320, 320), Image.Resampling.NEAREST).save(args.output / "mushroom-edited.png")
    (args.output / "mushroom-edited.js").write_text(recolored + "\n")
    report = {
        "source": "hand_authored_reference_programs",
        "purpose": "renderer_and_metric_smoke_test_not_model_performance",
        **evaluate(rows, {r["id"]: r["reference_code"] for r in rows}),
        "recolor_check": check_recolor(example["reference_code"], recolored, example["color_hex"], "#b8579e"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
