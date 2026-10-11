import numpy as np


from fingerprint import analyze


def test_cluster_interval_preserves_perfect_and_failed_predictions():
    rows = [{'prompt_id': 'a', 'label': 0}, {'prompt_id': 'a', 'label': 1}, {'prompt_id': 'b', 'label': 2}]
    assert analyze.accuracy_interval(rows, [[0, 1, 2], [0, 1, 2]], draws=100) == [1.0, 1.0]
    assert analyze.accuracy_interval(rows, [[1, 2, 0]], draws=100) == [0.0, 0.0]


def test_style_features_have_declared_counts():
    features = analyze.style_features('## Title\n- two words\n```python\nx=1\n```')
    np.testing.assert_allclose(features, np.log1p([8, 4, 2, 2]))
