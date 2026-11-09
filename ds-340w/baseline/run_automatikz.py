import argparse
import json
import time
from pathlib import Path

import torch
from automatikz.infer import TikzGenerator, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--prompt',
        default=(
            'A simple directed graph with three circular nodes labeled A, B, '
            'and C, and arrows from A to B and from B to C.'
        ),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model_id = 'nllg/tikz-clima-7b'
    started = time.time()

    print(f'Loading {model_id}', flush=True)
    model, tokenizer = load(model_id, device_map='auto')
    generator = TikzGenerator(model, tokenizer, stream=True)
    generator.pipeline.model = model

    print(f'Prompt: {args.prompt}', flush=True)
    document = generator(args.prompt, max_new_tokens=900)
    document.save(str(args.output / 'generated.tex'))
    report = {
        'model': model_id,
        'seed': args.seed,
        'prompt': args.prompt,
        'elapsed_seconds': round(time.time() - started, 2),
        'gpu': torch.cuda.get_device_name(),
        'generated_characters': len(document.code),
    }
    (args.output / 'generation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
