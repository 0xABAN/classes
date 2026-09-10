# The bias–variance tradeoff

Seven slides for CMPSC 448. Designed for a short, example-first explanation.

- **Present:** `bias-variance-tradeoff.pptx` (16:9; editable text, vector plots, and table).
- **Backup:** `bias-variance-tradeoff.pdf` (slides only).
- **Speaker notes:** included on every PowerPoint slide.
- **Rebuild:** `uv run build_deck.py` regenerates the PowerPoint and runs its checks.
  Export a new PDF after rebuilding; the script does not update the PDF.

Slide 1 compares straight and wiggly fits on identical training points from
one quadratic relationship. Slides 2 and 3 keep those fits frozen and show the
same 25 unseen observations with matching layouts. Slide 4 first reveals the
good-fit model family through repeated samples shared across all three degrees.
Slide 5 shows the balanced degree-2 fit on training data and then freezes it
for evaluation on the same unseen points. Slide 6 compares measured mean
squared errors for degrees 1, 2, and 12 on the original training set and all
1,000 test points. Slides 2, 3, and 5 display the first 25 of those test points.
Seeds are fixed; all examples are synthetic. Gaussian noise has standard
deviation 0.4 throughout training, testing, and repeated samples. Notes explain
the sampling setup, generalization, and validation/test distinction.

Visual reference: the saved Copycat `superlogical` profile. Uses Arial, original
plots, warm paper, dark ink, and a small original gradient accent; no source-site
logos, copy, portraits, or proprietary fonts. Rendered through LibreOffice for
visual inspection; PowerPoint-specific rendering has not been tested.
