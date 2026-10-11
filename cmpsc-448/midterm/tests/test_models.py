import pytest
import torch


@pytest.mark.parametrize('architecture', ['cnn', 'lstm'])
def test_padding_does_not_change_predictions_and_gradients_flow(architecture):
    from fingerprint.models import Classifier
    model = Classifier(architecture, vocab_size=20, embedding_dim=8, hidden=8).eval()
    short = torch.tensor([[2, 3, 4, 5, 6, 0]])
    padded = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0, 0]])
    length = torch.tensor([5])
    actual = model(short, length)
    assert actual.shape == (1, 3)
    torch.testing.assert_close(actual, model(padded, length), atol=1e-6, rtol=1e-5)
    torch.nn.functional.cross_entropy(actual, torch.tensor([1])).backward()
    assert model.embedding.weight.grad.abs().sum() > 0


def test_fast_recurrent_representation_matches_packed_bidirectional_reference():
    from fingerprint.models import Classifier
    model = Classifier('lstm', 20, embedding_dim=8, hidden=8).eval()
    assert hasattr(model, 'forward_lstm'), 'length-aware dense LSTM path is not implemented'
    reference = torch.nn.LSTM(8, 8, batch_first=True, bidirectional=True)
    weights = {}
    for name, parameter in model.forward_lstm.state_dict().items():
        weights[name] = parameter
    for name, parameter in model.backward_lstm.state_dict().items():
        weights[name + '_reverse'] = parameter
    reference.load_state_dict(weights)
    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]])
    lengths = torch.tensor([4, 2])
    embeddings = model.embedding(tokens)
    packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
    _, (hidden, _) = reference(packed)
    expected = model.output(torch.cat([hidden[-2], hidden[-1]], dim=1))
    torch.testing.assert_close(model(tokens, lengths), expected, atol=1e-6, rtol=1e-5)
