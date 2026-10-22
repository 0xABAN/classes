"""Prepare one downloaded book and sample strictly within each text split."""

import hashlib
import json
from pathlib import Path
import re
import urllib.request

import torch

ROOT = Path(__file__).resolve().parent.parent
SOURCE = 'https://www.gutenberg.org/cache/epub/55/pg55.txt'
SHA256 = '969bffab7740d4d8a0bac332d78bd0152ad4a40a2efc52f06c74a9bb6120be75'


def extract_story(raw):
    """Remove the license/front matter; preserve the chapter text and case."""
    start = re.search(r'^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*', raw, re.M)
    end = re.search(r'^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*', raw, re.M)
    if not start or not end:
        raise ValueError('Expected Project Gutenberg start/end markers')
    body = raw[start.end():end.start()]
    chapter = re.search(r'^Chapter I\s*\nThe Cyclone', body, re.M)
    if not chapter:
        raise ValueError('Cannot locate the first chapter')
    body = body[chapter.start():].replace('[Illustration]', '')
    return re.sub(r'\n{3,}', '\n\n', body).strip() + '\n'


class Tokenizer:
    """One character per token; unseen prompt characters raise an explicit error."""

    def __init__(self, characters):
        self.characters = ''.join(sorted(set(characters)))
        self.indices = {char: index for index, char in enumerate(self.characters)}

    def encode(self, text):
        unknown = sorted(set(text) - self.indices.keys())
        if unknown:
            raise ValueError(f'Unsupported characters: {unknown!r}')
        return [self.indices[char] for char in text]

    def decode(self, tokens):
        return ''.join(self.characters[index] for index in tokens)


def batch(tokens, size, context, generator):
    """Return windows and their one-character-shifted targets from one split."""
    if len(tokens) <= context:
        raise ValueError('Text split must be longer than the context window')
    starts = torch.randint(len(tokens) - context, (size,), generator=generator)
    x = torch.stack([tokens[i:i + context] for i in starts])
    y = torch.stack([tokens[i + 1:i + context + 1] for i in starts])
    return x, y


def prepare():
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    path = directory / 'pg55.txt'
    if not path.exists():
        urllib.request.urlretrieve(SOURCE, path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Source checksum changed; inspect the new edition before training')
    story = extract_story(raw.decode('utf-8-sig').replace('\r\n', '\n'))
    boundary = int(len(story) * 0.9)
    training, validation = story[:boundary], story[boundary:]
    tokenizer = Tokenizer(training)
    # Fail if validation contains unseen characters; never silently drop them.
    train_ids = torch.tensor(tokenizer.encode(training), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(validation), dtype=torch.long)
    metadata = {
        'title': 'The Wonderful Wizard of Oz', 'author': 'L. Frank Baum',
        'source_url': SOURCE, 'source_sha256': SHA256,
        'story_sha256': hashlib.sha256(story.encode()).hexdigest(),
        'characters': len(story), 'words': len(story.split()),
        'train_characters': len(training), 'validation_characters': len(validation),
        'split': 'Contiguous first 90% training; final 10% validation, before window sampling',
        'vocabulary_size': len(tokenizer.characters), 'vocabulary_fitted_on': 'training only',
    }
    (directory / 'story.txt').write_text(story)
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return train_ids, val_ids, tokenizer, metadata


if __name__ == '__main__':
    print(json.dumps(prepare()[3], indent=2))
