"""Aggregate the fixed experiment grid, including prompt-cluster uncertainty."""

from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy_interval(rows, predictions, draws=2000):
    """Bootstrap prompts, keeping their responses and all seeds together.

    This estimates uncertainty across held-out prompts conditional on the one
    training split. It is not an interval over new training corpora or LLMs.
    """
    labels = np.array([row['label'] for row in rows])
    correctness = (np.asarray(predictions) == labels).mean(axis=0)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['prompt_id']].append(index)
    group_indices = list(groups.values())
    totals = np.array([correctness[indices].sum() for indices in group_indices])
    counts = np.array([len(indices) for indices in group_indices])
    random = np.random.default_rng(2026)
    sampled = random.integers(0, len(groups), size=(draws, len(groups)))
    values = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def style_features(text):
    """Full-response log counts: words, newlines, list/headings, code fences."""
    return np.log1p([
        len(text.split()), text.count('\n'),
        len(re.findall(r'^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', text, flags=re.M)),
        text.count('```'),
    ])


def main():
    rows = json.loads(Path('data/selected.json').read_text())
    grouped = defaultdict(list)
    expected = {(a, c, seed) for a in ('cnn', 'lstm') for c in ('input', 'output', 'both', 'normalized') for seed in (11, 29, 47)}
    actual = set()
    for path in sorted(Path('results').glob('*_*_*.json')):
        result = json.loads(path.read_text())
        key = result['architecture'], result['condition']
        actual.add((*key, result['seed']))
        grouped[key].append(result)
    if actual != expected:
        raise ValueError(f'Incomplete experiment grid: missing {expected - actual}; extra {actual - expected}')

    summaries = []
    for (architecture, condition), runs in sorted(grouped.items()):
        for split in ('test', 'game'):
            test_rows = [row for row in rows if row['split'] == split]
            for stripped in ([False, True] if condition == 'output' else [False]):
                key = f'{split}_stripped' if stripped else split
                scores = [run[key] for run in runs]
                summaries.append({
                    'architecture': architecture, 'condition': 'stripped_test_only' if stripped else condition,
                    'split': split, 'n': len(test_rows),
                    'accuracy_mean': float(np.mean([s['accuracy'] for s in scores])),
                    'accuracy_sd': float(np.std([s['accuracy'] for s in scores], ddof=1)),
                    'f1_mean': float(np.mean([s['macro_f1'] for s in scores])),
                    'f1_sd': float(np.std([s['macro_f1'] for s in scores], ddof=1)),
                    'accuracy_prompt_ci': accuracy_interval(test_rows, [s['predictions'] for s in scores]),
                    'confusion_sum': np.sum([s['confusion_matrix'] for s in scores], axis=0).tolist(),
                })

    baseline = {}
    training_rows = [r for r in rows if r['split'] == 'train']
    train_features = np.array([style_features(row['response']) for row in training_rows])
    for name, columns in [('length', [0]), ('style', [0, 1, 2, 3])]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train_features[:, columns], [r['label'] for r in training_rows])
        baseline[name] = {}
        for split in ('test', 'game'):
            selected = [r for r in rows if r['split'] == split]
            predicted = model.predict(np.array([style_features(r['response']) for r in selected])[:, columns])
            labels = [r['label'] for r in selected]
            baseline[name][split] = {'accuracy': accuracy_score(labels, predicted),
                                     'macro_f1': f1_score(labels, predicted, average='macro'),
                                     'predictions': predicted.tolist(),
                                     'confusion_matrix': confusion_matrix(labels, predicted).tolist()}

    descriptive = []
    for split in ('train', 'test', 'game'):
        for label in range(3):
            selected = [r for r in rows if r['split'] == split and r['label'] == label]
            descriptive.append({'split': split, 'label': label, 'n': len(selected),
                                'median_words': float(np.median([len(r['response'].split()) for r in selected])),
                                'median_newlines': float(np.median([r['response'].count('\n') for r in selected])),
                                'source_counts': {s: sum(r['source'] == s for r in selected) for s in ('evol_instruct', 'ultrachat')}})
    report = {'summaries': summaries, 'baselines': baseline, 'descriptive': descriptive}
    Path('results/summary.json').write_text(json.dumps(report, indent=2))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for axis, architecture in zip(axes, ('cnn', 'lstm')):
        conditions = ['input', 'output', 'both']
        for offset, split, color, label in [(-0.18, 'test', '#8aa8bd', 'General prompts'), (0.18, 'game', '#23786d', 'Gaming prompts')]:
            values = [next(s for s in summaries if s['architecture'] == architecture and s['condition'] == c and s['split'] == split) for c in conditions]
            means = np.array([s['accuracy_mean'] for s in values]) * 100
            errors = np.array([s['accuracy_sd'] for s in values]) * 100
            axis.bar(np.arange(3) + offset, means, width=0.34, color=color, label=label, yerr=errors, capsize=3)
        axis.axhline(100 / 3, color='#777777', linestyle=':', linewidth=1)
        axis.set(xticks=np.arange(3), xticklabels=['Prompt', 'Response', 'Both'], title=architecture.upper(), ylim=(0, 100))
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig('results/accuracy.png', dpi=220)
    plt.close(fig)
    print('Wrote results/summary.json and results/accuracy.png')


if __name__ == '__main__':
    main()
