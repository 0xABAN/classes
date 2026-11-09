# AutomaTikZ baseline

The AutomaTikZ 7B baseline ran on an NVIDIA A40 on November 9, 2026. Given a prompt for three circular nodes labeled A, B, and C, the model created TikZ code with arrows from A to B and B to C. The unchanged generated code compiled successfully with Tectonic. Loading and generation together took 274.96 seconds, including model downloads. This is a single inference example, not a reproduction of the paper's benchmark scores. Pixel-art model training and preference tuning have not been run.

## Reproduce the run

Use the RunPod image `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` with an NVIDIA A40. The tested environment uses Python 3.10 and PyTorch 2.1.0+cu118.

From this directory:

```sh
git clone https://github.com/potamides/AutomaTikZ.git /workspace/automatikz
git -C /workspace/automatikz checkout 06134b9bb5c2cc009000573b8972089d66ff04e9
python -m pip install -r requirements.txt
python -m pip install --no-deps -e /workspace/automatikz
python -u run_automatikz.py --output results
```

The runner uses the upstream loader and generator with `nllg/tikz-clima-7b`. It replaces the generator's compiled model wrapper with the loaded model for eager inference, avoiding compilation startup. It saves the generated source and run metadata separately. The seed defaults to 7; `--prompt` accepts another description.

Compile the output on a machine with Tectonic installed:

```sh
cd results
tectonic --untrusted generated.tex
```

## Saved result

- [Generated source](results/generated.tex)
- [Rendered figure](results/generated.pdf)
- [Run metadata](results/generation.json)

The prompt asks for three circular nodes and two directed edges. The rendered result contains all three labels and both requested arrows. One small graph cannot establish performance on complex figures or pixel art.

## Parent paper

[AutomaTikZ: Text-Guided Synthesis of Scientific Vector Graphics with TikZ (ICLR 2024)](https://openreview.net/forum?id=v3K5TVP8kZ), by Jonas Belouadi, Anne Lauscher, and Steffen Eger. [Paper PDF](https://arxiv.org/pdf/2310.00367).

## Background recording

`record_rex.py` captures one explicitly selected Rex terminal block at two frames per second and renders an MP4. It does not capture the desktop or switch focus. Local dependencies are Pillow, FFmpeg, and the Rex CLI.

```sh
python record_rex.py --session SESSION --block BLOCK \
  --output demo.mp4 --state recording-state --font /System/Library/Fonts/Menlo.ttc
```

Use a fresh state directory for each recording. The recorder creates `ready` when capture starts; the demo process should wait for it before running. Create `done` after the demo finishes to close the video. If the demo saves `result.png` in that directory, the recorder displays it above the terminal output. The directory also contains timestamped terminal captures and a preview frame for inspection. The recording stops with an error after 30 minutes if `done` never appears.
