"""Offline character completion from a saved checkpoint; no external API calls."""

import argparse
from pathlib import Path

import torch

from ozlm.data import ROOT, Tokenizer
from ozlm.model import Config, TinyLM


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = TinyLM(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['weights'])
    return model.eval(), Tokenizer(checkpoint['characters'])


@torch.inference_mode()
def generate(model, prompt, count=300, temperature=0.8, top_k=20, seed=42):
    """Sample on CPU with a local RNG, retaining only the latest context window."""
    if not prompt or temperature <= 0 or count < 0 or top_k < 1:
        raise ValueError('Use a nonempty prompt, positive temperature/top-k and nonnegative length')
    model.eval()
    tokens = list(prompt)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(count):
        context = torch.tensor([tokens[-model.config.context:]], dtype=torch.long)
        logits = model(context)[0, -1] / temperature
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        selected = torch.multinomial(values.softmax(-1), 1, generator=rng)
        tokens.append(indices[selected].item())
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/model.pt')
    parser.add_argument('--prompt', help='Omit to enter prompts interactively')
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = load_checkpoint(args.checkpoint)
    print('Oz character model: text completion, not an instruction-following chatbot.')
    print(f'Checkpoint loaded locally; context = {model.config.context} characters.')

    def complete(prompt):
        result = generate(model, tokenizer.encode(prompt), args.length, args.temperature, seed=args.seed)
        print('\n' + tokenizer.decode(result) + '\n')

    if args.prompt is not None:
        complete(args.prompt)
        return
    print('Type a story opening. Enter /quit to stop. Try: Dorothy said')
    while True:
        try:
            prompt = input('Prompt> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == '/quit':
            break
        try:
            complete(prompt)
        except ValueError as error:
            print(error)


if __name__ == '__main__':
    main()
