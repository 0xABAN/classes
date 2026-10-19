# Training AI to Pixelate with Code

Team 7 · Adam Torres Encarnacion · DS 340W · Fall 2026

The proposal uses people's preferences between rendered programs to tune a pixel-art code model. The working prototype builds a 120-scene dataset, renders a restricted drawing language, checks outputs, and exports explicitly labeled preference pairs. Model generation and preference training have **not** been run.

## Presentation

- [Six-slide PowerPoint](presentation/training-ai-to-pixelate.pptx), with speaker notes for approximately 10 minutes.
- [PDF backup](presentation/training-ai-to-pixelate.pdf).
- [Talk notes](presentation/speaker-notes.txt).
- [Assignment requirements and coverage](presentation/requirements.md).

## Run the prototype

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```sh
cd ds-340w
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m pixelcode demo --output runs/demo
uv run python -m pixelcode dataset --output runs/prompts.jsonl
```

The demo uses hand-authored reference programs. It tests the renderer and evaluation code, not model performance. It saves five drawings, their source programs, a recolored mushroom, and a report. The recolor check requires every target-color pixel to change and every other pixel to stay fixed. The checked-in [examples](examples/) were generated with this command.

To evaluate model candidates, supply JSONL rows with `id` and `code` fields. Include every ID in the chosen split exactly once. Invalid programs count as failures; missing or duplicate IDs cause an error.

```sh
uv run python -m pixelcode evaluate \
  --dataset data/prompts.jsonl \
  --predictions runs/candidates.jsonl \
  --split test
```

## Code and data

| Path | Purpose |
| --- | --- |
| `pixelcode/renderer.py` | Parse and render bounded rectangle calls without executing JavaScript |
| `pixelcode/dataset.py` | Build prompts, reference programs, and fixed scene splits |
| `pixelcode/evaluation.py` | Check syntax, palette, placement, and recoloring |
| `pixelcode/preferences.py` | Export human-labeled train pairs as prompt/chosen/rejected records |
| [`data/prompts.jsonl`](data/prompts.jsonl) | The original 120-scene dataset used by the prototype |
| [`data/README.md`](data/README.md) | Data origin, fields, limits, and split policy |
| [`experiment.md`](experiment.md) | Proposed method and evaluation protocol |

The three parent papers supply the code representation, inspectable checks, and pairwise learning ideas. Their published results are not results from this project. See [references](presentation/references.md).
