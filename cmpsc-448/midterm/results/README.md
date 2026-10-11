# Experiment records

`manifest.json` is the ordered dataset manifest. For each run's `test` or `game` prediction list, prediction index 0 corresponds to the first manifest row with that split, and so on. Labels are 0 = GPT-4, 1 = Llama 2 70B Chat, and 2 = Falcon 40B Instruct. `source_row` is zero-based in the pinned source JSONL file.

Run filenames identify architecture, training condition, and seed. `test_stripped` and `game_stripped` evaluate a response-only model after stripping its test inputs, without retraining. Runs named `normalized` train and evaluate on stripped inputs. Each run retains its validation history, chosen epoch, parameter count, wall time, confusion matrices, and individual predictions. Wall times are not a speed benchmark: some CNN and LSTM runs executed concurrently.

`summary.json` contains means, sample standard deviations across seeds, and 95% percentile bootstrap intervals for accuracy. The bootstrap samples whole prompt groups with replacement, keeps all responses in each sampled group, and averages correctness across the three seeds. There are 2,000 bootstrap draws with seed 2026. The intervals are conditional on the fixed training/validation split.

Length and style logistic regressions use full responses. The neural models use fixed token prefixes. Baseline comparisons therefore diagnose possible shortcuts; they do not compare models under identical information budgets.

`data.json` records source URLs, checksums, and split sizes. `vocabulary.json` is fitted on training text. `environment.json` records the software and hardware used. The source data and learned weights are reproducible local files and are not committed.
