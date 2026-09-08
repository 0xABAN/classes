# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch>=2.6,<3"]
# ///
"""Classify synthetic road obstacles with multinomial logistic regression.

Run with: uv run main.py
"""

import torch
from torch import nn

RANGES = {
    #              width (m)  height (m)  speed (m/s)
    "pedestrian": ((0.3, 0.9), (1.2, 2.0), (0.0, 3.0)),
    "bicycle":    ((0.4, 1.0), (1.0, 1.9), (1.0, 10.0)),
    "car":        ((1.4, 2.2), (1.2, 2.0), (0.0, 25.0)),
}
CLASSES = tuple(RANGES)
LOW, HIGH = torch.tensor(list(RANGES.values())).unbind(-1)


def make_data(per_class, generator):
    """Draw independent uniform features per class; return shuffled examples."""
    labels = torch.arange(len(CLASSES)).repeat_interleave(per_class)
    features = LOW[labels] + torch.rand(
        len(labels), 3, generator=generator
    ) * (HIGH - LOW)[labels]
    order = torch.randperm(len(labels), generator=generator)
    return features[order], labels[order]


def main():
    torch.manual_seed(448)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(448)
    train_x, train_y = make_data(800, generator)
    test_x, test_y = make_data(200, generator)

    # Fit scaling on training data only, avoiding test-set leakage.
    mean, std = train_x.mean(0), train_x.std(0)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    model = nn.Linear(3, len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()  # Takes logits; includes log-softmax.

    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        loss = loss_fn(model(train_x), train_y)
        predictions = model(test_x).argmax(1)
        accuracy = (predictions == test_y).float().mean().item()
        confusion = torch.bincount(
            test_y * len(CLASSES) + predictions, minlength=len(CLASSES) ** 2
        ).reshape(len(CLASSES), len(CLASSES))

    # Small regression check: beat chance comfortably and count every example.
    assert torch.isfinite(loss) and accuracy > 0.75
    assert torch.equal(confusion.sum(1), torch.full((3,), 200))
    print(f"Training examples: {len(train_y)} | Test examples: {len(test_y)}")
    print(f"Final training loss: {loss.item():.4f}")
    print(f"Test accuracy: {accuracy:.2%} | Chance baseline: {1 / 3:.2%}")
    print("Confusion matrix: rows = actual, columns = predicted")
    print(f"{'':>12}" + "".join(f"{name:>12}" for name in CLASSES))
    for name, row in zip(CLASSES, confusion.tolist()):
        print(f"{name:>12}" + "".join(f"{count:>12}" for count in row))


if __name__ == "__main__":
    main()
