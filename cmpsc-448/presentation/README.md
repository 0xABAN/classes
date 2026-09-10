# The bias–variance tradeoff

Nine minimal slides for CMPSC 448: a beginner-friendly fitting primer,
followed by a focus on bias and variance through repeated training.

- **Present:** `bias-variance-tradeoff.pptx` (16:9; editable text, points, and curves).
- **Backup:** `bias-variance-tradeoff.pdf` (slides only).
- **Speaker notes:** included on every slide.
- **Rebuild:** `uv run build_deck.py` regenerates the PowerPoint and checks its
  numerical claims and structure. Export a new PDF after rebuilding.

## Story

1. Define a model, underfitting, and overfitting using identical data.
2. Illustrative training/test loss percentages compare high bias, good fit, and high variance.
3. Degree-12 polynomial fits vary across four training samples.
4. Linear regression gives more stable predictions on those same four samples.
5. Eight refits show prediction spread: variance.
6. Stable straight fits still miss the true relationship: bias.
7. Quadratic fits are closer to truth and more stable than the flexible fits.
8. Expected squared prediction error = bias² + variance + noise.
9. Two questions: is the average prediction off, and how much does retraining change it?

## Data and interpretation

The original quadratic relationship and sample A are preserved. All models
use 20 equally spaced inputs in [-1, 1]; only Gaussian output noise changes
between samples, with standard deviation 0.4. All model degrees see the same
eight samples. The plotted orange averages are arithmetic means of eight fits,
not exact population expectations. Some flexible curves leave the common
vertical range and are clipped; axes are never rescaled by model.

The generator verifies relative prediction spread, relative squared bias, and
the finite-ensemble squared-error decomposition. Notes distinguish those
empirical illustrations from population definitions and explain the independent
fresh-noise assumption. The noise term in the equation is variance (0.16), not
standard deviation (0.4).

Slide 2's training/test percentages (30/32, 8/10, 1/25) are explicitly
illustrative, not measured from the graphs. They represent a hypothetical loss
normalized to a shared fixed reference, not accuracy or ordinary MSE percentages.
The native editable table illustrates why training loss alone can mislead.

## Design

Inspired by the saved Copycat `superlogical` profile: warm paper, dark ink,
restrained orange/lavender, and a narrow original gradient on every slide.
No headers or footer rules; slide numbers remain. Uses Arial and no source-site
logos, portraits, copy, or proprietary fonts. PDF rendered through LibreOffice;
PowerPoint-specific rendering has not been tested.
