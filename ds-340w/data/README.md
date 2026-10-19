# Pixel-code pilot dataset

[Download the JSONL](https://raw.githubusercontent.com/0xABAN/classes/main/ds-340w/data/prompts.jsonl)

The dataset has five objects (mushroom, tree, house, boat, and heart), six colors, and four positions on a 32 by 32 canvas. There are 120 scenes: 100 for training and 20 for testing. Each row contains a prompt, the permitted colors, a placement rectangle, and reference code.

All prompts and sprite templates come from `pixelcode/dataset.py`. No external images or human preference labels are included. The references show one possible drawing, not the only correct answer. The small vocabulary and repeated templates make this a compositional pilot, not a benchmark of general artistic ability.

## Fields

| Field | Meaning |
| --- | --- |
| `id`, `scene_id` | Unique object-color-position combination |
| `split` | Fixed `train` or `test` assignment |
| `prompt` | Full drawing request and output syntax |
| `object`, `color_name`, `color_hex`, `position` | Scene attributes |
| `bounds` | `[x0, y0, x1, y1]`; upper bounds are exclusive |
| `allowed_colors` | Main color, dark detail color, and background |
| `reference_code` | Hand-authored sprite program generated from a template |
| `reference_origin` | Explicit provenance of the program |

## Split policy

Keep every candidate and paraphrase for a scene in its existing split. Test reference code must not enter model prompts, training, or hyperparameter selection. The split holds out complete combinations, not entire object categories. It measures limited compositional generalization. It does not measure unfamiliar-object recognition.

## Drawing contract

Only `rect(x, y, width, height, "#rrggbb");` calls are allowed. Coordinates and dimensions are integers. Rectangles must have positive area and fit inside the 32 by 32 canvas. Programs are limited to 512 commands and 65,536 characters. The renderer rejects unsupported input rather than repairing or executing it. The background is `#f5f5f4`; later rectangles cover earlier ones.

Regenerate the file with `uv run python -m pixelcode dataset --output data/prompts.jsonl`. The output is deterministic. Reference images are provided in [`../examples`](../examples/).
