# /// script
# requires-python = ">=3.12"
# dependencies = ["python-pptx>=1.0,<2", "numpy>=2,<3", "pillow>=11,<13"]
# ///
"""Build the deck with `uv run build_deck.py`; plots and text stay editable."""

from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
PAPER = "F5F5F4"
INK = "181D16"
MUTED = "686C66"
RULE = "D4D6D0"
ORANGE = "C65326"
PURPLE = "7963AF"
FAINT_PURPLE = "C4B8DC"
FONT = "Arial"
NOISE_STD = .4


def text(slide, value, x, y, w, h, size=24, color=INK, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(0)
    return box


def line(slide, x1, y1, x2, y2, color=RULE, width=1, dashed=False):
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shape.line.color.rgb = RGBColor.from_string(color)
    shape.line.width = Pt(width)
    if dashed:
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return shape


def path(slide, points, color, width=2.5, dashed=False):
    """Keep each plotted curve a single editable PowerPoint vector shape."""
    coords = [(round(x * 1000), round(y * 1000)) for x, y in points]
    builder = slide.shapes.build_freeform(*coords[0], scale=Inches(1) / 1000)
    builder.add_line_segments(coords[1:], close=False)
    shape = builder.convert_to_shape()
    shape.fill.background()
    shape.line.color.rgb = RGBColor.from_string(color)
    shape.line.width = Pt(width)
    if dashed:
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return shape


def dot(slide, x, y, color=INK, hollow=False, r=.045):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(x-r), Inches(y-r), Inches(r*2), Inches(r*2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(PAPER if hollow else color)
    shape.line.color.rgb = RGBColor.from_string(color)
    shape.line.width = Pt(1.5)


def gradient():
    """An original soft color field; no source-site artwork or font assets."""
    stops = np.array([[196, 175, 240], [244, 167, 149], [247, 183, 108], [235, 207, 128]])
    yy = np.linspace(0, 3, 900)
    rgb = np.stack([np.interp(yy, np.arange(4), stops[:, c]) for c in range(3)], axis=1)
    pixels = np.repeat(rgb[:, None, :], 320, axis=1)
    pixels += np.random.default_rng(9).normal(0, 1.4, pixels.shape[:2])[:, :, None]
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def base(prs, n, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(PAPER)
    slide.shapes.add_picture(gradient(), Inches(12.87), Inches(.22), Inches(.24), Inches(7.06))
    text(slide, title, .62, 1.05, 12, .8, 38)
    text(slide, f"{n:02d}/07", 11.94, 7.12, .75, .22, 11, MUTED)
    return slide


def plot(slide, rect, curves=(), dots=(), ylim=(-.65, 2.45), xlim=(-1, 1),
         axis_labels=("x", "y")):
    """Map data into native shapes; split out-of-range curves instead of flattening them."""
    x, y, w, h = rect

    def project(a, b):
        return x + (a-xlim[0]) / (xlim[1]-xlim[0])*w, y+h-(b-ylim[0]) / (ylim[1]-ylim[0])*h

    line(slide, x, y+h, x+w, y+h)
    line(slide, x, y, x, y+h)
    text(slide, axis_labels[0], x+w-.1, y+h+.10, .7, .3, 14, MUTED)
    text(slide, axis_labels[1], x-.30, y-.2, 1, .3, 14, MUTED)
    for xx, yy, color, dashed, width in curves:
        segment = []
        for a, b in zip(xx, yy):
            if ylim[0] <= b <= ylim[1]:
                segment.append(project(a, b))
            else:
                if len(segment) > 1:
                    path(slide, segment, color, width, dashed)
                segment = []
        if len(segment) > 1:
            path(slide, segment, color, width, dashed)
    for xx, yy, color, hollow in dots:
        for a, b in zip(xx, yy):
            if xlim[0] <= a <= xlim[1] and ylim[0] <= b <= ylim[1]:
                dot(slide, *project(a, b), color, hollow)


def legend(slide, items, x=.92, y=6.45):
    for label, color, kind in items:
        if kind in ("dot", "hollow"):
            dot(slide, x+.07, y+.1, color, kind == "hollow")
        else:
            line(slide, x, y+.1, x+.28, y+.1, color, 2, kind == "dash")
        text(slide, label, x+.39, y-.03, 2.7, .35, 13, MUTED)
        x += 3.04


def note(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def data():
    # Every fit and the loss table share this one noisy quadratic dataset.
    x = np.linspace(-1, 1, 20)
    truth = lambda z: .3 + .4*z + 1.3*z*z
    y = truth(x) + np.random.default_rng(12).normal(0, NOISE_STD, len(x))
    xt = np.random.default_rng(41).uniform(-1, 1, 1000)
    yt = truth(xt) + np.random.default_rng(42).normal(0, NOISE_STD, len(xt))
    models = [Polynomial.fit(x, y, d) for d in (1, 2, 12)]
    losses = [(float(np.mean((p(x)-y)**2)), float(np.mean((p(xt)-yt)**2))) for p in models]
    return x, y, truth, xt, yt, models, losses


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
    prs.core_properties.title = "The bias–variance tradeoff"
    prs.core_properties.subject = "CMPSC 448 — an example-first introduction"
    prs.core_properties.author = "CMPSC 448"
    x, y, truth, xt, yt, models, losses = data()
    g = np.linspace(-1, 1, 400)
    panels = [(left, 3.1, 3.35, 2.85) for left in (.98, 5.02, 9.06)]
    opening_panels = [(left, 3.1, 5.0, 2.85) for left in (.98, 7.12)]
    # Both unseen-data slides show the same subset of the loss table's test set.
    fresh_x, fresh_y = xt[:25], yt[:25]

    s = base(prs, 1, "Which model would you trust?")
    for rect, model, label in zip(opening_panels, (models[0], models[2]), ("Straight line", "Wiggly curve")):
        text(s, label, rect[0], 2.65, 3.4, .3, 16)
        plot(s, rect, [(g, model(g), ORANGE, False, 3)], [(x, y, INK, False)])
    legend(s, [("training observations", INK, "dot"), ("fitted model", ORANGE, "line")])
    note(s, f"Ask: Which fit looks convincing, and which would predict new observations well? Pause before naming the concepts. Both models use exactly the same 20 training observations and identical axis limits. They are degree-1 and degree-12 polynomial fits. The underlying relationship is quadratic. Do not reveal the degree-2 good fit yet; it appears in the repeated-sample comparison on slide 4. All examples use x in [-1, 1], independent Gaussian noise with standard deviation {NOISE_STD}, and arbitrary units. Curves and dots are editable vector objects.")

    s = base(prs, 2, "Too simple to learn the pattern.")
    plot(s, (.98, 2.94, 7.12, 3.10),
         [(g, models[0](g), ORANGE, False, 3)],
         [(fresh_x, fresh_y, PURPLE, True)])
    text(s, "High bias", 8.73, 3.52, 3.8, .7, 30, ORANGE)
    text(s, "A straight line cannot\nlearn this curve.", 8.75, 4.32, 3.65, 1.2, 23)
    legend(s, [("unseen observations", PURPLE, "hollow"), ("unchanged model", ORANGE, "line")])
    note(s, "Keep the straight fit from slide 1 frozen. Reveal the first 25 observations from the independent 1,000-point test set; slide 3 will show exactly these same observations and axis limits. No model is refitted, and there is no distribution shift or extrapolation. The line still misses the curvature. Bias is the systematic difference between the average prediction over repeated training samples and the true regression function. Here the model family only permits straight lines, so even its average fit cannot capture curvature. The true curve is deliberately hidden here, as it would be in practice; only the frozen model and unseen observations are shown. Underfitting usually leaves substantial error even on training data. More samples alone will not make a straight model quadratic. Linear regression can also use polynomial features: it is the straight-line specification here that is restrictive.")

    s = base(prs, 3, "Too complex to learn the pattern.")
    plot(s, (.98, 2.94, 7.12, 3.10),
         [(g, models[2](g), ORANGE, False, 3)],
         [(fresh_x, fresh_y, PURPLE, True)])
    text(s, "High variance", 8.73, 3.52, 4, .7, 30, ORANGE)
    text(s, "Small changes in data.\nBig changes in the fit.", 8.75, 4.32, 3.8, 1.2, 23)
    legend(s, [("unseen observations", PURPLE, "hollow"), ("unchanged model", ORANGE, "line")])
    note(s, "Keep the degree-12 fit from slide 1 frozen and reveal the exact same 25 unseen observations as slide 2. The model was trained on the original 20 points, not these new points. Its wiggles follow noise from training that is not repeated in the test sample. There is no distribution shift or extrapolation. After revealing the good-fit alternative on slide 4, use slide 6 to quantify why the lowest training loss does not guarantee the lowest test loss. Variance means variability of model predictions across possible training datasets, not variability of the input data and not simply a curve looking wiggly. One fit suggests noise-chasing; slide 4 demonstrates sensitivity through repeated samples. Flexibility is not inherently bad: its risk depends on sample size, noise, regularization, and the learning procedure.")

    s = base(prs, 4, "A closer look at variance")
    ensembles = [[], [], []]
    for seed in range(20, 28):
        sample_y = truth(x) + np.random.default_rng(seed).normal(0, NOISE_STD, len(x))
        for curves, degree in zip(ensembles, (1, 2, 12)):
            model = Polynomial.fit(x, sample_y, degree)
            curves.append((g, model(g), FAINT_PURPLE, False, 1.25))
    for rect, curves, label in zip(panels, ensembles, ("Underfit", "Good fit", "Overfit")):
        text(s, label, rect[0], 2.65, 3.4, .3, 16)
        plot(s, rect, curves + [(g, truth(g), INK, True, 2)])
    legend(s, [("one fit per sample", PURPLE, "line"), ("true pattern", INK, "dash")])
    note(s, "Generate eight fresh training samples at the original 20 x positions. For each sample, all three model degrees receive exactly the same noisy outputs. This is the first reveal of the degree-2 good-fit model family. Each pale curve is one refit; the original fits from slides 1–3 are not used here. Straight fits cluster but miss the quadratic truth. Degree-2 curves stay close to the truth. Degree-12 curves spread out, especially near the boundaries. All three panels share identical axes. Some degree-12 fits leave the visible vertical range; those portions are omitted rather than compressed. This controlled repeated-sampling comparison isolates sensitivity to the training sample. Bias and variance are defined at an input x and can vary across the domain.")

    s = base(prs, 5, "A closer fit is not always a better model.")
    c = np.linspace(0, 1, 220)
    train = .10 + .48*np.exp(-4*c)
    test = .28 + 1.65*(c-.48)**2
    assert np.all(test > train)
    plot(s, (1.02, 3.0, 7.2, 2.83),
         [(c, train, MUTED, False, 2.5), (c, test, ORANGE, False, 3)],
         ylim=(0, 1.08), xlim=(0, 1), axis_labels=("", "loss"))
    text(s, "simpler", 1.02, 6.0, 2, .3, 14, MUTED)
    text(s, "more flexible →", 6.52, 6.0, 2.4, .3, 14, MUTED)
    text(s, "Underfit", 1.15, 2.6, 2, .35, 16, MUTED)
    text(s, "Balance", 4.0, 2.6, 2, .35, 16, INK)
    text(s, "Overfit", 6.74, 2.6, 2, .35, 16, MUTED)
    text(s, "Choose using\nvalidation data.", 9.05, 3.72, 3.5, 1.4, 28)
    legend(s, [("training loss", MUTED, "line"), ("expected test loss", ORANGE, "line")])
    text(s, "Schematic, not measured", 9.04, 6.44, 3.4, .3, 12, MUTED)
    note(s, "This is the classic schematic, not the measured curve from our examples and not a universal law. In nested least-squares model families training loss cannot increase as flexibility grows, while expected test loss may first improve and then worsen. Under squared error, expected prediction error at x decomposes into squared bias, variance, and irreducible noise, assuming a stable data-generating process and averaging over training samples and fresh output noise. There is no guarantee of a smooth U-shape in every regime. Choose hyperparameters using a validation split or cross-validation; reserve a final test set for an honest, one-time evaluation. Do not tune on test results.")

    s = base(prs, 6, "The smallest training loss can lose.")
    # Native PowerPoint table: all labels and measured values can be edited.
    table = s.shapes.add_table(4, 4, Inches(.86), Inches(3.1), Inches(11.57), Inches(2.55)).table
    for col, width in zip(table.columns, [3.45, 2.72, 2.70, 2.70]):
        col.width = Inches(width)
    rows = [
        ["Fit", "Model degree", "Training MSE ↓", "Testing MSE ↓"],
        ["Underfitting", "1", f"{losses[0][0]:.3f}", f"{losses[0][1]:.3f}"],
        ["Good fit", "2", f"{losses[1][0]:.3f}", f"{losses[1][1]:.3f}"],
        ["Overfitting", "12", f"{losses[2][0]:.3f}", f"{losses[2][1]:.3f}"],
    ]
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string("E8E3EF" if r == 2 else PAPER)
            cell.margin_left, cell.margin_top = Inches(.18), Inches(.16)
            cell.margin_right, cell.margin_bottom = Inches(.12), Inches(.08)
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(16 if r == 0 else 24)
                p.font.color.rgb = RGBColor.from_string(MUTED if r == 0 else INK)
                p.font.bold = r == 2
            # Override theme borders with quiet horizontal rules only.
            tc_pr = cell._tc.get_or_add_tcPr()
            for edge in ("lnL", "lnR", "lnT", "lnB"):
                el = OxmlElement(f"a:{edge}")
                el.set("w", "6350" if edge == "lnB" else "0")
                fill = OxmlElement("a:solidFill" if edge == "lnB" else "a:noFill")
                if edge == "lnB":
                    color = OxmlElement("a:srgbClr")
                    color.set("val", RULE)
                    fill.append(color)
                el.append(fill)
                tc_pr.append(el)
    note(s, f"All rows are actual computed results on the shared quadratic process: y = 0.3 + 0.4x + 1.3x² + noise, with noise standard deviation {NOISE_STD}. Training uses 20 equally spaced x values on [-1, 1]; test uses 1,000 independently drawn uniform x values in that same range and fresh noise. Degrees 1, 2, and 12 were fixed for this demonstration, not selected by searching test loss. Mean squared error is the average of (prediction minus observed y)²; lower is better and units are squared output units. The underfit has relatively high losses on both sets. The degree-2 fit has low, similar losses. Degree 12 achieves the lowest training error, but has higher test error than degree 2. Test loss need not exceed training loss for every finite sample; note that degree 1's test loss is lower here. A single gap does not by itself prove high variance; repeated sampling supplies the conceptual evidence. The degree-1 and degree-12 rows are the frozen models shown on slides 1–3. The degree-2 row is fitted to the same original training set; slide 4 introduced this model family through fresh refits. All rows use identical training and test data. Slides 2 and 3 display only the first 25 test points for readability, while the table evaluates all 1,000.")

    s = base(prs, 7, "Learn the pattern.\nNot the noise.")
    # Give the two-line closing headline more room than the standard title.
    for shape in s.shapes:
        if shape.has_text_frame and shape.text == "Learn the pattern.\nNot the noise.":
            shape.height = Inches(1.65)
            for p in shape.text_frame.paragraphs:
                p.font.size = Pt(48)
    for label, body, xx in [
        ("BIAS", "Too rigid to capture\nthe relationship.", .67),
        ("VARIANCE", "Too sensitive to\nthe training sample.", 4.77),
        ("GENERALIZATION", "How well it predicts\nnew observations.", 8.85),
    ]:
        line(s, xx, 4.4, xx+3.45, 4.4)
        text(s, label, xx, 4.66, 3.5, .3, 12, MUTED)
        text(s, body, xx, 5.12, 3.6, .95, 23)
    text(s, "Tune on validation. Report on the untouched test set.", .67, 6.38, 11.9, .4, 23, ORANGE)
    note(s, "Close by asking: If training loss is tiny but validation loss is much larger, what might you change? Possible answers include reducing complexity, adding regularization, or collecting more representative training data. If both losses are poor, consider better features or a more expressive model, while also checking data quality and optimization. These are diagnostic clues, not absolute rules. Training learns parameters; validation chooses settings; test estimates performance after decisions are frozen. Teaching background: An Introduction to Statistical Learning, 2nd edition, section 2.2.2, The Bias-Variance Trade-Off, https://www.statlearning.com/. All numerical data and graphics here are original synthetic examples. Visual language is inspired by the saved superlogical profile; no brand assets are reused.")

    # Native shapes inherit a shadow from the template unless the effect is disabled.
    for slide in prs.slides:
        for effect in slide._element.xpath(".//a:effectRef"):
            effect.set("idx", "0")

    # Small runnable checks protect both the teaching claim and the saved artifact.
    assert losses[2][0] < losses[1][0] < losses[0][0]
    assert losses[1][1] < losses[2][1] < losses[0][1]
    assert len(prs.slides) == 7
    assert all(slide.has_notes_slide for slide in prs.slides)
    output = ROOT / "bias-variance-tradeoff.pptx"
    prs.save(output)
    reopened = Presentation(output)
    assert len(reopened.slides) == 7
    saved_table = next(shape.table for shape in reopened.slides[5].shapes if shape.has_table)
    for row, pair in enumerate(losses, start=1):
        for column, loss in enumerate(pair, start=2):
            assert saved_table.cell(row, column).text == f"{loss:.3f}"
    # Check the saved points: shared training panels and identical unseen slides.
    opening_dots = [shape for shape in reopened.slides[0].shapes
                    if shape.name.startswith("Oval")][:2*len(x)]
    assert len(opening_dots) == 2*len(x)
    for a, b in zip(opening_dots[:len(x)], opening_dots[len(x):]):
        assert abs((b.left-a.left) / Inches(1) - 6.14) < 2e-6
        assert a.top == b.top
    unseen_positions = []
    for index in (1, 2):
        dots = [shape for shape in reopened.slides[index].shapes
                if shape.name.startswith("Oval")][:len(fresh_x)]
        assert len(dots) == len(fresh_x)
        unseen_positions.append([(shape.left, shape.top) for shape in dots])
    assert unseen_positions[0] == unseen_positions[1]
    assert all("Simple curve" not in shape.text for shape in reopened.slides[0].shapes
               if shape.has_text_frame)
    for number, slide in enumerate(reopened.slides, start=1):
        visible_text = [shape.text for shape in slide.shapes if shape.has_text_frame]
        assert f"{number:02d}/07" in visible_text
        assert any(shape.left == Inches(12.87) and shape.width == Inches(.24)
                   for shape in slide.shapes)
        assert not any(shape.top == Inches(7.02) for shape in slide.shapes)
        if number in (2, 3):
            assert "true pattern" not in visible_text
            assert len(slide._element.xpath(".//a:custGeom")) == 1
        assert not any("BIAS–VARIANCE" in value for value in visible_text)
        assert not any("Two different datasets" in value for value in visible_text)
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= prs.slide_width + Inches(.02)
            assert shape.top + shape.height <= prs.slide_height + Inches(.02)
    print(f"Saved and structurally checked: {output}")
    print("Measured (training, testing) MSE:", losses)


if __name__ == "__main__":
    build()
