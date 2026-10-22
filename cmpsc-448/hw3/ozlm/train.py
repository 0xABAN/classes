"""Train next-character prediction from random weights and retain best validation loss."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ozlm.data import ROOT, batch, prepare
from ozlm.model import Config, TinyLM
from ozlm.generate import generate, load_checkpoint


@torch.inference_mode()
def evaluate(model, tokens, device, batches=16):
    """Use fixed sampled windows for comparable checks, without consuming training RNG."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(2026)
    losses = []
    for _ in range(batches):
        x, y = batch(tokens, 16, model.config.context, generator)
        logits = model(x.to(device))
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten()).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--eval-batches', type=int, default=16)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.eval_batches) < 1:
        parser.error('Step counts and evaluation intervals must be positive')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available; choose --device cpu explicitly')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'model.pt').exists():
        parser.error('Output already contains a model; use a new --output directory')

    torch.set_num_threads(4)
    torch.manual_seed(448)
    train, validation, tokenizer, data_metadata = prepare()
    config = Config(vocab_size=len(tokenizer.characters))
    model = TinyLM(config)
    prompt = 'Dorothy said'
    untrained = tokenizer.decode(generate(model, tokenizer.encode(prompt), count=200, seed=42))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    sampler = torch.Generator().manual_seed(448)
    history = []
    best_loss, best_step = float('inf'), 0
    start = time.perf_counter()
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters; {len(train):,} training characters; device={args.device}', flush=True)

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            training_loss = evaluate(model, train, args.device, args.eval_batches)
            validation_loss = evaluate(model, validation, args.device, args.eval_batches)
            history.append({'step': step, 'train_loss': training_loss, 'validation_loss': validation_loss})
            print(f'step {step:4}: train {training_loss:.4f}, validation {validation_loss:.4f}, elapsed {time.perf_counter()-start:.1f}s', flush=True)
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                torch.save({
                    'config': asdict(config), 'characters': tokenizer.characters,
                    'weights': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    'step': step, 'validation_loss': validation_loss,
                }, args.output / 'model.pt')
        if step == args.steps:
            break
        model.train()
        # Warm up, then cosine decay. Only the fixed step budget controls the schedule.
        progress = step / max(1, args.steps - 1)
        learning_rate = 0.0001 + 0.0009 * (1 + math.cos(math.pi * progress)) / 2
        learning_rate *= min(1.0, (step + 1) / min(100, args.steps))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        x, y = batch(train, 24, config.context, sampler)
        logits = model(x.to(args.device))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(args.device).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - start
    record = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'pretrained_weights': False, 'seed': 448, 'device': args.device,
        'torch_version': str(torch.__version__), 'config': asdict(config),
        'parameters': sum(p.numel() for p in model.parameters()),
        'steps_completed': args.steps, 'batch_size': 24, 'seconds': elapsed,
        'optimizer': 'AdamW', 'weight_decay': 0.01, 'learning_rate_peak': 0.001,
        'learning_rate_floor': 0.0001, 'warmup_steps': min(100, args.steps),
        'gradient_clip': 1.0, 'best_step': best_step,
        'best_validation_loss': best_loss, 'best_validation_perplexity': math.exp(best_loss),
        'evaluation': 'Fixed random validation windows; used for checkpoint selection, not an independent test',
        'evaluation_batches': args.eval_batches, 'data': data_metadata, 'history': history,
    }
    (args.output / 'training.json').write_text(json.dumps(record, indent=2) + '\n')
    restored, tokenizer = load_checkpoint(args.output / 'model.pt')
    samples = {'untrained': {'prompt': prompt, 'output': untrained}, 'trained': [],
               'settings': {'seed': 42, 'temperature': 0.8, 'top_k': 20, 'new_characters': 200}}
    for prompt in ['Dorothy said', 'The Scarecrow', 'The Wizard of Oz', 'Once upon a time']:
        output = tokenizer.decode(generate(restored, tokenizer.encode(prompt), count=200, seed=42))
        samples['trained'].append({'prompt': prompt, 'output': output})
    (args.output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    print(f'Saved best step {best_step}; validation loss {best_loss:.4f}; training {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
