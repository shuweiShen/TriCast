"""Windows-safe single-device TriCast training, evaluation and staged checks."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.data_factory import data_dict
from models.TriCast import Model


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='JSON experiment defaults; CLI arguments override these')
    p.add_argument('--llm_name', default='gpt2')
    p.add_argument('--llm_registry', default=None)
    p.add_argument('--llm_dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    p.add_argument('--mode', choices=['smoke', 'train', 'eval', 'inspect'], default='smoke')
    p.add_argument('--tricast_stage', type=int, choices=range(6), default=5)
    p.add_argument('--llm_path', default=None)
    p.add_argument('--llm_layers', type=int, default=None, help='Omit/null for all pretrained layers')
    p.add_argument('--seq_len', type=int, default=192)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--patch_len', type=int, default=96)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--num_experts', type=int, default=4)
    p.add_argument('--msdr_scales', type=int, nargs='+', default=[16, 32, 64])
    p.add_argument('--msdr_max_lag', type=int, default=8)
    p.add_argument('--msdr_rho_floor', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--data', choices=[k for k in data_dict if k != 'm4'], default='ETTh1')
    p.add_argument('--root_path', default='./dataset/ETT-small')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--features', choices=['S', 'M', 'MS'], default='M')
    p.add_argument('--target', default='OT')
    p.add_argument('--freq', default='h')
    p.add_argument('--percent', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action='store_true', help='CUDA bfloat16 autocast')
    p.add_argument('--max_batches', type=int, default=0, help='Debug cap per split; 0 means all')
    p.add_argument('--output_dir', default='./checkpoints/TriCast')
    p.add_argument('--checkpoint', default=None, help='Required for eval')
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text(encoding='utf-8-sig'))
        actions = {a.dest: a for a in p._actions}
        for key, value in defaults.items():
            if key not in actions or key in ('help', 'config'):
                p.error(f'Unknown config key: {key}')
            action = actions[key]
            if value is None:
                if key not in ('llm_layers', 'llm_path', 'llm_registry', 'checkpoint'):
                    p.error(f'{key} cannot be null')
            elif action.choices and value not in action.choices:
                p.error(f'Invalid config {key}: {value}')
            elif isinstance(action, argparse._StoreTrueAction) and not isinstance(value, bool):
                p.error(f'{key} must be a JSON boolean')
            elif action.type:
                try:
                    defaults[key] = ([action.type(v) for v in value] if action.nargs == '+' else action.type(value))
                except (TypeError, ValueError):
                    p.error(f'Invalid config value for {key}')
        p.set_defaults(**defaults)
    return p.parse_args(argv)


class ObservedWindows(Dataset):
    """Reuse original train-only scaler/splits, return all variables per window.

    The upstream __getitem__ flattens variables into independent samples.
    """
    def __init__(self, args, split):
        self.base = data_dict[args.data](root_path=args.root_path, data_path=args.data_path,
            flag=split, size=[args.seq_len, 0, args.pred_len], features=args.features,
            target=args.target, timeenc=1, freq=args.freq, percent=args.percent)
        self.seq_len, self.pred_len = args.seq_len, args.pred_len
        if len(self) <= 0:
            raise ValueError('Not enough observations for the requested windows')

    def __len__(self):
        return len(self.base.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        end = i + self.seq_len
        return (self.base.data_x[i:end].astype(np.float32),
                self.base.data_y[end:end + self.pred_len].astype(np.float32))


def context(args):
    if args.amp and args.device.startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def evaluate(model, loader, args, target_index):
    model.eval()
    squared = absolute = count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(args.device), y.to(args.device)
            with context(args):
                prediction = model(x)
            if args.features == 'MS':
                prediction, y = prediction[..., target_index], y[..., target_index]
            error = prediction.float() - y
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += error.numel()
    if not count:
        raise ValueError('Empty evaluation loader')
    return {'mse': squared / count, 'mae': absolute / count, 'elements': count}


def smoke(model, args, batch):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    x, y = (t.to(args.device) for t in batch)
    before = model.to_llm.weight.detach().clone()
    with context(args):
        prediction = model(x)
        loss = (prediction.float() - y).square().mean()
    assert prediction.shape == y.shape and torch.isfinite(prediction).all()
    if args.tricast_stage >= 1:
        assert model.last_cp.shape == x.shape
        assert torch.isfinite(model.last_cp).all()
        assert ((model.last_cp >= 0) & (model.last_cp <= 1)).all()
    loss.backward()
    gradients = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    assert model.to_llm.weight.grad.abs().sum() > 0, 'No gradient through frozen LLM'
    assert all(p.grad is None for p in model.llm_model.parameters())
    if args.tricast_stage >= 5:
        assert model.cp_router.router.weight.grad.abs().sum() > 0
        assert torch.allclose(model.cp_router.last_weights.sum(-1), torch.ones(x.size(0) * args.enc_in, device=x.device))
    optimizer.step()
    assert not torch.equal(before, model.to_llm.weight)
    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x)
        assert torch.allclose(a, b), 'Evaluation is not deterministic'
    return {'stage': args.tricast_stage, 'loss': loss.item(), 'shape': list(prediction.shape),
            'gradient_tensors': len(gradients), 'status': 'passed'}


def main():
    args = arguments()
    if args.max_batches < 0 or not 1 <= args.percent <= 100:
        raise ValueError('Invalid max_batches or percent')
    if args.batch_size < 1 or args.train_epochs < 1 or args.patience < 1:
        raise ValueError('batch_size, train_epochs and patience must be positive')
    if args.mode == 'inspect':
        from utils.llm_backbone import inspect_backbone
        _, _, identity = inspect_backbone(args)
        windows = {split: len(ObservedWindows(args, split)) for split in ('train', 'val', 'test')}
        print(json.dumps({'backbone': identity, 'windows': windows, 'config': vars(args)}, indent=2))
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = ObservedWindows(args, 'train')
    args.enc_in = train.base.data_x.shape[-1]
    target_index = -1
    if args.features == 'MS':
        import pandas as pd
        columns = list(pd.read_csv(Path(args.root_path) / args.data_path, nrows=0).columns[1:])
        # Custom loaders move the target column to the end; ETT keeps CSV order.
        target_index = columns.index(args.target) if args.data.startswith('ETT') else -1
    model = Model(args).to(args.device)
    print('LLM: ' + json.dumps(model.backbone_identity), flush=True)
    print(f'Stage {args.tricast_stage}; device={args.device}; channels={args.enc_in}; '
          f'trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.mode == 'smoke':
        result = smoke(model, args, next(iter(loader)))
        (output / f'smoke_stage{args.tricast_stage}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    validation = DataLoader(ObservedWindows(args, 'val'), batch_size=args.batch_size, num_workers=args.num_workers)
    test = DataLoader(ObservedWindows(args, 'test'), batch_size=args.batch_size, num_workers=args.num_workers)
    config = vars(args).copy()
    config_file = 'eval_config.json' if args.mode == 'eval' else 'resolved_config.json'
    (output / config_file).write_text(json.dumps(config, indent=2), encoding='utf-8')
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / 'best.pt'
    if args.mode == 'train':
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate, weight_decay=args.weight_decay)
        best, stale, history = float('inf'), 0, []
        for epoch in range(args.train_epochs):
            model.train()
            total, count = 0., 0
            started = time.time()
            for i, (x, y) in enumerate(loader):
                if args.max_batches and i >= args.max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                x, y = x.to(args.device), y.to(args.device)
                with context(args):
                    prediction = model(x)
                    if args.features == 'MS':
                        prediction, y = prediction[..., target_index], y[..., target_index]
                    loss = (prediction.float() - y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                total += loss.item() * y.numel()
                count += y.numel()
                if (i + 1) % 50 == 0:
                    print(f'epoch={epoch + 1} batch={i + 1} loss={loss.item():.6f}', flush=True)
            metrics = evaluate(model, validation, args, target_index)
            row = {'epoch': epoch + 1, 'train_mse': total / count, 'validation': metrics,
                   'seconds': time.time() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics['mse'] < best:
                best, stale = metrics['mse'], 0
                adapters = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('llm_model.')}
                torch.save({'adapters': adapters, 'config': config, 'epoch': epoch + 1,
                            'backbone_identity': model.backbone_identity}, checkpoint)
            else:
                stale += 1
            (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if stale >= args.patience:
                break
    elif not args.checkpoint:
        raise ValueError('--checkpoint is required for eval')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'backbone_identity' in saved:
        if saved['backbone_identity'] != model.backbone_identity:
            raise ValueError('Checkpoint LLM identity differs from the selected local backbone')
    elif args.llm_name != 'gpt2':
        raise ValueError('Legacy checkpoint has no LLM identity; it can only be loaded with gpt2')
    else:
        print('Legacy checkpoint: LLM identity was not recorded; verify original GPT-2 weights.', flush=True)
    for key in ['seq_len', 'pred_len', 'patch_len', 'd_model', 'd_ff', 'llm_layers', 'tricast_stage',
                'num_experts', 'enc_in', 'n_heads', 'msdr_scales', 'msdr_max_lag', 'msdr_rho_floor',
                'dropout', 'data', 'data_path', 'freq', 'features', 'target', 'percent']:
        # null and an explicit full-depth setting describe the same architecture.
        old = saved['config'][key]
        current = config[key]
        if key == 'llm_layers':
            old = old or model.backbone_identity['full_layers']
            current = current or model.backbone_identity['full_layers']
        if old != current:
            raise ValueError(f'Checkpoint configuration differs: {key}')
    incompatible = model.load_state_dict(saved['adapters'], strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('llm_model.') for k in incompatible.missing_keys)
    result = {'test': evaluate(model, test, args, target_index), 'debug_max_batches': args.max_batches,
              'checkpoint': str(checkpoint), 'config': config}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
