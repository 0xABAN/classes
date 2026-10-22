import pytest
import torch


def test_book_extraction_excludes_gutenberg_and_front_matter():
    from ozlm.data import extract_story
    raw = 'license\n*** START OF THE PROJECT GUTENBERG EBOOK OZ ***\nContents\n Chapter I. The Cyclone\n\nChapter I\nThe Cyclone\n\nDorothy lived.\n[Illustration]\n\n*** END OF THE PROJECT GUTENBERG EBOOK OZ ***\nlicense'
    story = extract_story(raw)
    assert story.startswith('Chapter I\nThe Cyclone')
    assert 'Dorothy lived.' in story
    assert 'license' not in story and 'Illustration' not in story
    with pytest.raises(ValueError):
        extract_story('not a Gutenberg book')


def test_batch_targets_shift_one_character_and_stay_in_split():
    from ozlm.data import batch
    tokens = torch.arange(100)
    x, y = batch(tokens, 8, 16, torch.Generator().manual_seed(1))
    assert x.shape == y.shape == (8, 16)
    assert torch.equal(x + 1, y)
    assert y.max() < 100
    with pytest.raises(ValueError):
        batch(torch.arange(16), 1, 16, torch.Generator())


def test_tokenizer_roundtrip_and_unknown_character_error():
    from ozlm.data import Tokenizer
    tokenizer = Tokenizer(' abDory\n')
    assert tokenizer.decode(tokenizer.encode('Dor\n')) == 'Dor\n'
    with pytest.raises(ValueError, match='Unsupported'):
        tokenizer.encode('🙂')


def test_attention_cannot_see_future_characters():
    from ozlm.model import TinyLM, Config
    torch.manual_seed(1)
    model = TinyLM(Config(vocab_size=10, width=16, heads=2, layers=2, context=8)).eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 8, 9]])
    torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])
    assert model(a).shape == (1, 5, 10)


def test_model_learns_repeating_sequence():
    from ozlm.model import TinyLM, Config
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = TinyLM(Config(vocab_size=3, width=16, heads=2, layers=1, context=12, dropout=0))
    x = torch.tensor([[0, 1, 2] * 4] * 4)
    y = (x + 1) % 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.1


def test_generation_keeps_prompt_and_handles_long_context():
    from ozlm.model import TinyLM, Config
    from ozlm.generate import generate
    model = TinyLM(Config(vocab_size=5, width=16, heads=2, layers=1, context=8)).eval()
    prompt = [1, 2] * 8
    a = generate(model, prompt, count=10, temperature=0.8, top_k=3, seed=42)
    b = generate(model, prompt, count=10, temperature=0.8, top_k=3, seed=42)
    assert a == b and a[:16] == prompt and len(a) == 26
    with pytest.raises(ValueError):
        generate(model, [], count=10)
    with pytest.raises(ValueError):
        generate(model, [1], temperature=0)


def test_checkpoint_roundtrip(tmp_path):
    from ozlm.model import TinyLM, Config
    from ozlm.generate import load_checkpoint
    from dataclasses import asdict
    config = Config(vocab_size=3, width=16, heads=2, layers=1, context=8)
    model = TinyLM(config).eval()
    path = tmp_path / 'model.pt'
    torch.save({'config': asdict(config), 'characters': 'abc', 'weights': model.state_dict()}, path)
    restored, tokenizer = load_checkpoint(path)
    x = torch.tensor([[0, 1]])
    torch.testing.assert_close(restored(x), model(x))
    assert tokenizer.decode([0, 1]) == 'ab'


def test_evaluation_is_repeatable_and_restores_training_mode():
    from ozlm.train import evaluate
    from ozlm.model import TinyLM, Config
    model = TinyLM(Config(vocab_size=3, width=16, heads=2, layers=1, context=8))
    tokens = torch.tensor([0, 1, 2] * 30)
    first = evaluate(model, tokens, 'cpu', batches=2)
    assert model.training
    assert first == evaluate(model, tokens, 'cpu', batches=2)


def test_training_command_saves_loadable_checkpoint(tmp_path):
    import subprocess, sys, json
    from ozlm.generate import load_checkpoint
    subprocess.run([sys.executable, '-m', 'ozlm.train', '--steps', '2', '--eval-every', '1', '--eval-batches', '1', '--output', str(tmp_path)], check=True, capture_output=True)
    model, tokenizer = load_checkpoint(tmp_path / 'model.pt')
    record = json.loads((tmp_path / 'training.json').read_text())
    assert record['steps_completed'] == 2
    assert record['pretrained_weights'] is False
    assert model.config.vocab_size == len(tokenizer.characters)
