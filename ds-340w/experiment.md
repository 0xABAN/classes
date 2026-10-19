# Proposed experiment

Compare Qwen2.5-Coder-7B-Instruct before and after token-level DPO with LoRA. For each training prompt, generate two programs and collect an explicit human choice between valid, different renderings. Keep ties in the annotation log. The unchanged base model is the frozen reference. This is a proposal; candidates, preference labels, and a trained adapter have not been produced.

The renderer rejects unsupported syntax and out-of-bounds rectangles. Log invalid generations before filtering for preference training. The pair exporter accepts only training scenes, records the reviewer ID, and rejects identical renderings. Technical checks cannot establish whether a drawing depicts the requested object.

## Comparison

Evaluate both models on the 20 held-out prompts with three fixed seeds (17, 29, and 43), giving 60 comparisons. Use the same prompt format, sampling settings, and output limits. Fix training settings before testing; do not choose a checkpoint by test performance. Candidate sampling temperature is 0.7, top-p is 0.9, and the maximum output length is 2,048 tokens for both models. The pilot training configuration will be fixed before labels are used; no test-based hyperparameter search is planned.

Invalid code is a failure and stays in the denominator. A valid output beats an invalid output in the usability comparison; two invalid outputs tie. For valid pairs, seek three independent raters, hide model names, and randomize display order. Preserve each vote rather than negotiating a consensus. Use majority choice; no majority is a tie. Report the actual number of raters and their disagreements.

| Measure | Definition |
| --- | --- |
| Validity | Accepted programs / all generated programs |
| Palette | Allowed colors and requested main color / all generations |
| Placement | Nonempty drawing entirely in requested quadrant / all generations |
| Prompt compliance | Rater judgment of requested object, color, and position; report separately |
| Appearance | Rater comparison without assuming prompt compliance |
| Preference credit | `(wins + 0.5 × ties) / 60`; report wins, losses, and ties too |
| Edit success | Recolor all target pixels and leave every other pixel unchanged; invalid or empty edits fail |

Resample whole prompts when estimating a 95 percent interval, retaining all three seed comparisons in each sampled prompt. The seeds do not make 60 independent prompts. The proposed targets are at least 95 percent validity and 60 percent preference credit, without lower validity or prompt compliance than the base model. Call the preference result inconclusive if its interval includes 50 percent.

## Limits and next work

The templates are simple, the dataset is small, and ratings may reflect a narrow set of tastes. A small number of non-tied training pairs may not improve the model. DPO can favor appealing outputs that miss the prompt, which is why the metrics stay separate. Training requires compute that has not been allocated for this pilot. No paid run is started by the prototype commands.

The next work is model candidate generation, independent preference annotation, adapter training, and the held-out comparison. The checked-in report verifies hand-authored reference programs only. It is not evidence that a trained model meets these targets.
