"""Deterministic, prompt-disjoint sampling and train-only text encoding."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

MODELS = ('gpt-4', 'llama-2-70b-chat', 'falcon-40b-instruct')
REVISION = '40b436560ca83a8dba36114c22ab3c66e43f6d5e'
SOURCES = {
    'evol_instruct': '1845a636816a6e713ca8f508372d4386cb82c8864b047da89c57ed9d1e3fa76a',
    'ultrachat': '1955e9268ae30eeb414068c2fe482def21a76cbfa6b0a63f4319a8ab19f900ff',
}
GAME_CANDIDATE = re.compile(r'\b(?:games?|gaming|gameplay|gamers?|videogames?|video games?|board games?|role.playing|dungeons? and dragons?|d&d|NPCs?|MMORPG|RPG|Minecraft|chess|poker|tic.tac.toe)\b', re.I)
IDENTITY = re.compile(r'\b(?:chatgpt|gpt[ -]?(?:3\.5|4)(?:-turbo)?|llama(?:[ -]?2)?(?:-70b-chat)?|falcon(?:-40b-instruct)?|openai|meta|tii)\b', re.I)
TOKEN = re.compile(r'<model>|\n|\w+|[^\w\s]', re.UNICODE)


def mask_identity(text):
    """Replace an explicit, predeclared list of model/vendor names in every arm."""
    return IDENTITY.sub('<model>', text)


def normalize(text):
    """Strip common Markdown markers and collapse whitespace; preserve word case."""
    text = re.sub(r'^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)', '', text, flags=re.M)
    text = text.replace('`', '').replace('**', '').replace('__', '')
    return ' '.join(text.split())


def prompt_key(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def select_rows(source_rows, train_per_class=800, val_per_class=150):
    """Reserve every complete three-model prompt for the matched test set.

    Duplicate prompts are consolidated by normalized prompt hash. Other prompt
    groups enter train/validation before class balancing, so sampling cannot
    put two responses to the same prompt on opposite sides of the split.
    """
    groups = defaultdict(dict)
    for item in source_rows:
        key = prompt_key(item['instruction'])
        for completion in item['completions']:
            if completion['model'] not in MODELS or not completion['response'].strip():
                continue
            label = MODELS.index(completion['model'])
            groups[key].setdefault(label, {
                'prompt_id': key, 'source_row': item['source_row'],
                'source': item['source'], 'label': label,
                'model': completion['model'],
                'prompt': mask_identity(item['instruction']),
                'response': mask_identity(completion['response']),
            })

    matched = {key for key, answers in groups.items() if len(answers) == 3}
    remaining = sorted(set(groups) - matched)
    random.Random(2026).shuffle(remaining)
    validation = set(remaining[:max(1, len(remaining) // 5)])
    pools = defaultdict(list)
    for key, answers in sorted(groups.items()):
        split = 'test' if key in matched else 'val' if key in validation else 'train'
        for label, row in sorted(answers.items()):
            pools[split, label].append({**row, 'split': split})

    selected = []
    for split, count in [('train', train_per_class), ('val', val_per_class), ('test', len(matched))]:
        for label in range(3):
            candidates = pools[split, label]
            random.Random(2026 + label).shuffle(candidates)
            if len(candidates) < count:
                raise ValueError(f'{split}/{MODELS[label]}: need {count}, found {len(candidates)}')
            selected.extend(candidates[:count])
    if not matched:
        raise ValueError('No complete matched test prompts')
    return selected


def select_domains(source_rows, game_review, train_per_class=800, val_per_class=150):
    """Keep reviewed game prompts out of all model fitting and selection.

    Broad keyword candidates that failed manual review are excluded too. This
    avoids training on ambiguous game material such as gamified math puzzles.
    Game membership is defined only from prompts, before inspecting scores.
    """
    source_rows = list(source_rows)
    game_keys = {prompt_key(row['instruction']) for row in source_rows
                 if row['source_row'] in game_review[row['source']]}
    excluded_keys = game_keys | {prompt_key(row['instruction']) for row in source_rows
                                 if GAME_CANDIDATE.search(row['instruction'])}
    general = [row for row in source_rows if prompt_key(row['instruction']) not in excluded_keys]
    selected = select_rows(general, train_per_class, val_per_class)
    pools = defaultdict(dict)
    for row in source_rows:
        key = prompt_key(row['instruction'])
        if key not in game_keys:
            continue
        for completion in row['completions']:
            if completion['model'] in MODELS and completion['response'].strip():
                label = MODELS.index(completion['model'])
                pools[label].setdefault(key, {
                    'prompt_id': key, 'source': row['source'], 'source_row': row['source_row'],
                    'label': label, 'model': completion['model'], 'split': 'game',
                    'prompt': mask_identity(row['instruction']),
                    'response': mask_identity(completion['response']),
                })
    count = min(len(pools[label]) for label in range(3))
    if count == 0:
        raise ValueError('Game holdout must contain all three labels')
    for label in range(3):
        candidates = [value for _, value in sorted(pools[label].items())]
        random.Random(2026 + label).shuffle(candidates)
        selected.extend(candidates[:count])
    return selected


def tokenize(text):
    return TOKEN.findall(text)


def build_vocab(rows, limit=10000):
    counts = Counter()
    for row in rows:
        if row['split'] == 'train':
            counts.update(tokenize(row['prompt']))
            counts.update(tokenize(row['response']))
    vocabulary = {'<pad>': 0, '<unk>': 1, '<sep>': 2}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit - 3]:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(text, vocabulary, limit):
    return [vocabulary.get(token, 1) for token in tokenize(text)[:limit]] or [1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/source'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    source_rows = []
    urls = {}
    for source, checksum in SOURCES.items():
        path = args.cache / f'{source}.jsonl'
        urls[source] = f'https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/{REVISION}/{source}.jsonl'
        if not path.exists():
            subprocess.run(['curl', '-L', '--fail', '--show-error', urls[source], '-o', str(path)], check=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f'Source checksum mismatch for {source}: {actual}')
        for index, line in enumerate(path.open()):
            source_rows.append({**json.loads(line), 'source': source, 'source_row': index})
    rows = select_domains(source_rows, json.loads(Path('game-review.json').read_text()))
    Path('data').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data/selected.json').write_text(json.dumps(rows))
    manifest = [{key: value for key, value in row.items() if key not in ('prompt', 'response')} for row in rows]
    Path('results/manifest.json').write_text(json.dumps(manifest, indent=2))
    metadata = {'source_urls': urls, 'sha256': SOURCES, 'models': MODELS,
                'counts': dict(Counter(row['split'] for row in rows)),
                'unique_prompts': {s: len({r['prompt_id'] for r in rows if r['split'] == s}) for s in ('train', 'val', 'test', 'game')}}
    Path('results/data.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
