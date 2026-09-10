# The bias–variance tradeoff

Eight slides for CMPSC 448. Designed for a short, example-first explanation.

- **Present:** `bias-variance-tradeoff.pptx` (16:9; editable text, vector plots, and table).
- **Backup:** `bias-variance-tradeoff.pdf` (slides only).
- **Speaker notes:** included on every PowerPoint slide.
- **Rebuild:** `uv run build_deck.py` regenerates the PowerPoint and runs its checks.
  Export a new PDF after rebuilding; the script does not update the PDF.

Slides 1–5 use one quadratic relationship. Degrees 1, 2, and 12 are fitted to
identical training points, then evaluated on identical unseen points. The
repeated-sample comparison also gives every model the same data within each
refit. Slide 6 is a schematic, not an empirical loss curve. Slide 7 reports
measured mean squared errors for the exact models shown on slides 1 and 4.
Slide 4 displays the first 25 of the table's 1,000 test points. Seeds are fixed;
all examples are synthetic. Notes explain the sampling setup, validation/test
distinction, and limits of the usual U-shaped curve.

Visual reference: the saved Copycat `superlogical` profile. Uses Arial, original
plots, warm paper, dark ink, and a small original gradient accent; no source-site
logos, copy, portraits, or proprietary fonts. Rendered through LibreOffice for
visual inspection; PowerPoint-specific rendering has not been tested.
