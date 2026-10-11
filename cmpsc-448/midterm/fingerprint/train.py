"""Train fixed CNN/LSTM configurations and save every held-out prediction."""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

from fingerprint.data import TOKEN, build_vocab, encode, normalize
from fingerprint.models import Classifier


def make_example(row, condition, vocabulary):
    prompt = encode(row['prompt'], vocabulary, 96)
    answer = row['response']
    if condition == 'normalized':
        # Strip only the same raw prefix visible to the unmodified classifier.
        # Otherwise deleting newlines could expose additional answer content.
        matches = list(TOKEN.finditer(answer))
        if len(matches) > 160:
            answer = answer[:matches[159].end()]
        answer = normalize(answer)
    response = encode(answer, vocabulary, 160)
    if condition == 'input':
        tokens = prompt
    elif condition in ('output', 'normalized'):
        tokens = response
    elif condition == 'both':
        tokens = prompt + [2] + response
    else:
        raise ValueError(f'Unknown condition: {condition}')
    return tokens, row['label']


def collate(examples):
    sequences, labels = zip(*examples)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    tokens = torch.zeros(len(examples), int(lengths.max()), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = torch.tensor(sequence)
    return tokens, lengths, torch.tensor(labels)


def evaluate(model, examples):
    model.eval()
    predicted, actual = [], []
    with torch.no_grad():
        for tokens, lengths, labels in DataLoader(examples, batch_size=128, collate_fn=collate):
            predicted.extend(model(tokens, lengths).argmax(1).tolist())
            actual.extend(labels.tolist())
    return {
        'accuracy': accuracy_score(actual, predicted),
        'macro_f1': f1_score(actual, predicted, labels=[0, 1, 2], average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
    }, predicted


def fit(model, training, validation, epochs=12, patience=3, learning_rate=0.001, batch_size=64):
    """Select weights only by validation macro-F1, never by a test result."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate)
    history, best_score, best_epoch, best_weights = [], -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for tokens, lengths, labels in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tokens, lengths), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scores, _ = evaluate(model, validation)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_accuracy': scores['accuracy'], 'val_macro_f1': scores['macro_f1']})
        print(f'  epoch {epoch}: loss={np.mean(losses):.4f}, val_f1={scores["macro_f1"]:.4f}', flush=True)
        if scores['macro_f1'] > best_score:
            best_score, best_epoch = scores['macro_f1'], epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    return {'epochs': history, 'best_epoch': best_epoch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    parser.add_argument('--architectures', nargs='+', choices=['cnn', 'lstm'], default=['cnn', 'lstm'])
    parser.add_argument('--conditions', nargs='+', choices=['input', 'output', 'both', 'normalized'], default=['input', 'output', 'both', 'normalized'])
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows = json.loads(Path('data/selected.json').read_text())
    vocabulary = build_vocab(rows)
    Path('results').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results/vocabulary.json').write_text(json.dumps(vocabulary))

    for architecture in args.architectures:
        for condition in args.conditions:
            examples = {split: [make_example(row, condition, vocabulary) for row in rows if row['split'] == split] for split in ('train', 'val', 'test', 'game')}
            for seed in args.seeds:
                print(f'Start {architecture}/{condition}/seed={seed}', flush=True)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                model = Classifier(architecture, len(vocabulary))
                start = time.perf_counter()
                training = fit(model, examples['train'], examples['val'], epochs=args.epochs)
                result = {'architecture': architecture, 'condition': condition, 'seed': seed,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'training': training, 'seconds': time.perf_counter() - start,
                          'torch_version': torch.__version__, 'device': 'cpu'}
                for split in ('test', 'game'):
                    if not examples[split]:
                        continue
                    scores, predictions = evaluate(model, examples[split])
                    result[split] = {**scores, 'predictions': predictions}
                    if condition == 'output':
                        stripped = [make_example(row, 'normalized', vocabulary) for row in rows if row['split'] == split]
                        scores, predictions = evaluate(model, stripped)
                        result[f'{split}_stripped'] = {**scores, 'predictions': predictions}
                name = f'{architecture}_{condition}_{seed}'
                torch.save(model.state_dict(), f'checkpoints/{name}.pt')
                Path(f'results/{name}.json').write_text(json.dumps(result, indent=2))
                print(json.dumps({k: result[k] for k in ('architecture', 'condition', 'seed', 'seconds')}),
                      {split: round(result[split]['accuracy'], 3) for split in ('test', 'game') if split in result}, flush=True)


if __name__ == '__main__':
    main()
