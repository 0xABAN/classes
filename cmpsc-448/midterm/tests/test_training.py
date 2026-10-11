import torch


from fingerprint import train


def test_conditions_keep_response_budget_and_ignore_answer_for_input():
    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2, 'P': 3, 'R': 4}
    row = {'prompt': ' '.join(['P'] * 200), 'response': ' '.join(['R'] * 300), 'label': 1}
    assert train.make_example(row, 'input', vocab)[0] == [3] * 96
    assert train.make_example(row, 'output', vocab)[0] == [4] * 160
    assert train.make_example(row, 'both', vocab)[0] == [3] * 96 + [2] + [4] * 160
    tokens, lengths, labels = train.collate([([3, 4], 0), ([4], 1)])
    assert tokens.tolist() == [[3, 4], [4, 0]]
    assert lengths.tolist() == [2, 1]
    assert labels.tolist() == [0, 1]


def test_training_can_learn_a_tiny_separable_corpus(tmp_path):
    from fingerprint.models import Classifier
    torch.set_num_threads(1)
    torch.manual_seed(7)
    examples = [([3 + label] * 5, label) for label in range(3) for _ in range(12)]
    model = Classifier('cnn', 8, embedding_dim=8, hidden=8)
    history = train.fit(model, examples, examples, epochs=20, patience=20, learning_rate=0.02, batch_size=36)
    scores, predictions = train.evaluate(model, examples)
    assert scores['accuracy'] > 0.95
    assert history['best_epoch'] >= 1
    assert len(predictions) == 36


def test_normalization_cannot_expose_additional_response_content():
    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2, 'R': 3, '\n': 4, 'HIDDEN': 5}
    row = {'prompt': 'P', 'response': ('R\n' * 80) + 'HIDDEN', 'label': 0}
    normalized, _ = train.make_example(row, 'normalized', vocab)
    assert 5 not in normalized
    assert normalized == [3] * 80
