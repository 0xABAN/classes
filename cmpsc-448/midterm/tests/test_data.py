from collections import Counter


from fingerprint import data


def test_formatting_ablation_keeps_words_and_sentence_punctuation():
    assert data.normalize('## Heading\n\n- **Hello**, world!\n2. `Goodbye`.') == 'Heading Hello, world! Goodbye.'
    assert data.normalize(data.normalize('## Heading\n- item')) == 'Heading item'


def test_identity_masking_is_case_insensitive():
    assert data.mask_identity('GPT-4 and LLaMA 2 and Falcon, by OpenAI.') == '<model> and <model> and <model>, by <model>.'


def test_matched_test_and_disjoint_prompt_groups():
    rows = []
    for i in range(80):
        completions = [{'model': name, 'response': f'Answer {i} from a system.', 'principle': 'helpfulness'} for name in data.MODELS]
        # Only two prompts contain every target model.
        if i >= 2:
            completions = completions[:2] if i % 2 else completions[1:]
        rows.append({'source': 'toy', 'source_row': i, 'instruction': f'Question {i}', 'completions': completions})
    selected = data.select_rows(rows, train_per_class=5, val_per_class=2)
    split_ids = {s: {r['prompt_id'] for r in selected if r['split'] == s} for s in ('train', 'val', 'test')}
    assert not split_ids['train'] & split_ids['val']
    assert not split_ids['train'] & split_ids['test']
    assert not split_ids['val'] & split_ids['test']
    assert Counter(r['label'] for r in selected if r['split'] == 'test') == {0: 2, 1: 2, 2: 2}
    assert len([r for r in selected if r['split'] == 'train']) == 15


def test_vocabulary_never_learns_validation_text():
    rows = [{'split': 'train', 'prompt': 'ordinary', 'response': 'words'}, {'split': 'val', 'prompt': 'secretvalidation', 'response': 'secretanswer'}]
    vocab = data.build_vocab(rows)
    assert 'ordinary' in vocab
    assert 'secretvalidation' not in vocab
    assert 'secretanswer' not in vocab
    assert data.encode('secretanswer', vocab, 4) == [1]


def test_game_prompts_never_enter_training_or_general_test():
    rows = []
    for i in range(80):
        completions = [{'model': name, 'response': 'A response.', 'principle': 'helpfulness'} for name in data.MODELS]
        if i >= 5:
            completions = completions[:2] if i % 2 else completions[1:]
        rows.append({'source': 'toy', 'source_row': i, 'instruction': f'Question {i}', 'completions': completions})
    selected = data.select_domains(rows, {'toy': [0, 1, 2]}, train_per_class=5, val_per_class=2)
    game_ids = {r['prompt_id'] for r in selected if r['split'] == 'game'}
    other_ids = {r['prompt_id'] for r in selected if r['split'] != 'game'}
    assert len(game_ids) == 3
    assert not game_ids & other_ids
    assert len([r for r in selected if r['split'] == 'game']) == 9


def test_missing_source_metadata_is_rejected():
    import pytest
    row = {'instruction': 'Question', 'completions': [{'model': name, 'response': 'Answer'} for name in data.MODELS]}
    with pytest.raises(KeyError):
        data.select_rows([row], train_per_class=0, val_per_class=0)
