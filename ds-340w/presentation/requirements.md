# Midterm presentation coverage

Source: [Week 9: In-class Midterm Presentation](https://psu.instructure.com/courses/2484449/assignments/18675220), checked October 19, 2026. Canvas lists the upload deadline as October 23 at 11:59 p.m. EDT. This does not replace the assigned in-class presentation date.

The assignment asks for approximately ten minutes: five on background, the problem, and three parent papers; five on the proposed method and evaluation. Six slides keep the visuals short. The speaker notes carry the explanation.

| Requirement | Where to find it |
| --- | --- |
| Title, team, motivation, applications, outline | Slide 1 and its notes |
| Three parent papers and their connections | Slide 2; linked paper titles; `references.md` |
| Problem, research question, hypothesis | Slide 3 |
| Proposed approach and justification | Slide 4; `../experiment.md` |
| Dataset, split, baseline, evaluation | Slide 5; `../data/README.md` |
| Success criteria, limits, conclusion | Slide 6 and its notes |
| Accessible GitHub repository with development code | [Project folder](https://github.com/0xABAN/classes/tree/main/ds-340w) |
| Dataset link | [JSONL download](https://raw.githubusercontent.com/0xABAN/classes/main/ds-340w/data/prompts.jsonl) |
| PowerPoint file | [training-ai-to-pixelate.pptx](training-ai-to-pixelate.pptx) |

## Timing

Slides 1–3: 100, 130, and 70 seconds. Slides 4–6: 120, 120, and 60 seconds. Each half totals five minutes. Rehearse once; the notes are a guide, not a requirement to read every word.

## Rebuild

From `ds-340w/`:

```sh
uv run --group presentation python presentation/build_deck.py
soffice --headless --convert-to pdf --outdir presentation presentation/training-ai-to-pixelate.pptx
uv run python -m unittest discover -s tests -v
```

The build uses the supplied PowerPoint package as `template.pptx`, preserving its page size, background, theme, and right-edge gradient. Text and diagrams remain editable. The PDF is a viewing backup; the PPTX is the requested presentation artifact.

## Status boundaries

The checked-in code renders programs, builds the dataset, checks outputs, and exports labeled training pairs. The example report uses hand-authored reference programs. It is not a model evaluation. Model candidate generation, preference collection, and adapter training remain the proposed experiment.

Publishing these files does not submit anything to Canvas. The assignment accepts the repository link, dataset link, and PPT file through submission or comments.
