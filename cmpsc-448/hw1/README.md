# HW1: classifying road obstacles

## Run

```sh
cd cmpsc-448/hw1
uv run main.py
```

`uv` installs Python and the script's PyTorch dependency in an isolated cache.
No dataset download, GPU, NumPy operations, or training framework is needed.
PyTorch may warn that NumPy is unavailable; this script does not use its NumPy
integration.

## Problem and data

Predict whether a road obstacle is a **pedestrian**, **bicycle (with rider)**,
or **car**, given its width, height, and speed. These are assumed measurements,
not features extracted from images.

Each feature is sampled independently from a continuous uniform distribution
within the following class-specific bounds. These are invented ranges for a
classroom experiment, not measured population statistics.

| Class | Width (m) | Height (m) | Speed (m/s) |
| --- | --- | --- | --- |
| Pedestrian | 0.3–0.9 | 1.2–2.0 | 0–3 |
| Bicycle | 0.4–1.0 | 1.0–1.9 | 1–10 |
| Car | 1.4–2.2 | 1.2–2.0 | 0–25 |

The script generates 800 training and 200 test examples per class: 2,400
training examples and 600 held-out test examples. Both sets are separate draws
from the same generator, seeded with 448. Class counts are equal, giving a
33.33% chance baseline. Features are standardized using only training means
and standard deviations.

## Model

Multinomial logistic regression computes three scores as `Wx + b`, then
interprets their softmax as class probabilities. `nn.Linear(3, 3)` supplies
these scores with 12 trainable parameters: nine weights and three biases.
There are no hidden layers.

`CrossEntropyLoss` applies log-softmax internally, so it receives raw scores
rather than probabilities. Adam minimizes the mean training cross-entropy
for 500 full-batch steps with learning rate 0.05. Prediction chooses the
largest score. The test set is used only for evaluation, not optimization.

## Observed performance

A local CPU run produced:

```text
Training examples: 2400 | Test examples: 600
Final training loss: 0.1961
Test accuracy: 93.50% | Chance baseline: 33.33%
Confusion matrix: rows = actual, columns = predicted
              pedestrian     bicycle         car
  pedestrian         191           9           0
     bicycle          30         170           0
         car           0           0         200
```

That is 561 correct predictions out of 600. Recall is 95.5% for pedestrians,
85.0% for bicycles, and 100% for cars. All 39 errors confuse pedestrians and
bicycles, whose feature ranges overlap. Cars are especially easy because their
width range is disjoint from the other classes. Results may vary slightly
across PyTorch versions and platforms.

## Limitations and design choices

- Scaling prevents the larger numerical speed range from dominating training
  solely because of its units. Fitting the scaler on training data avoids
  leaking test statistics.
- Overlapping pedestrian and bicycle distributions make some observations
  ambiguous even with perfect measurements. A linear classifier also cannot
  represent every boundary of these class-specific uniform distributions.
- Balanced synthetic classes, independent features, and noiseless measurements
  are simplifications. High accuracy here is not evidence of road safety or
  performance on real sensor data.

## Report

[report.md](report.md) covers the problem, data, model, measured results,
implementation decisions, and lessons. It includes [results.png](results.png),
a browser screenshot of the script's captured output.

Review the report before submitting. PDF export remains pending. To export
with Pandoc and Tectonic installed, run from this directory:

```sh
pandoc report.md --pdf-engine=tectonic -o report.pdf
```
