"""Run ETT experiments sequentially; dry-run never loads weights or datasets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['gpt2-xl', 'gpt2-large', 'gpt2', 'qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'tinyllama-1.1b']
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']


def commands(args):
    for model in args.models:
        for dataset in args.datasets:
            config_path = ROOT / 'configs' / 'experiments' / f'{model}_{dataset}.json'
            config = json.loads(config_path.read_text(encoding='utf-8'))
            for horizon in args.pred_lens:
                depth = args.llm_layers if args.llm_layers is not None else config['llm_layers']
                run_id = f'{model}/{dataset}/sl{config["seq_len"]}_pl{horizon}_layers{depth or "full"}/{args.run_tag}'
                output = Path(args.output_root).resolve() / run_id
                common = [sys.executable, str(ROOT / 'run_tricast.py'), '--config', str(config_path),
                          '--pred_len', str(horizon), '--output_dir', str(output)]
                for key in ['batch_size', 'train_epochs', 'llm_layers', 'max_batches', 'device', 'llm_path']:
                    value = getattr(args, key)
                    if value is not None:
                        common += ['--' + key, str(value)]
                if args.amp:
                    common += ['--amp']
                modes = ['train', 'eval'] if args.mode == 'train-test' else [
                    'eval' if args.mode == 'test' else args.mode]
                for mode in modes:
                    command = common + ['--mode', mode]
                    if mode == 'eval':
                        command += ['--checkpoint', str(output / 'best.pt')]
                    yield model, dataset, horizon, mode, command


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'test', 'train-test', 'smoke', 'inspect'], default='train-test')
    p.add_argument('--models', choices=MODELS, nargs='+', default=MODELS)
    p.add_argument('--datasets', choices=DATASETS, nargs='+', default=DATASETS)
    p.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720])
    p.add_argument('--output_root', default=str(ROOT / 'checkpoints' / 'tricast_matrix'))
    p.add_argument('--run_tag', default='default', help='Use a new tag for changed settings/debug runs')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--batch_size', type=int)
    p.add_argument('--train_epochs', type=int)
    p.add_argument('--llm_layers', type=int)
    p.add_argument('--max_batches', type=int)
    p.add_argument('--device')
    p.add_argument('--llm_path')
    p.add_argument('--amp', action='store_true')
    args = p.parse_args(argv)
    if args.llm_path and len(args.models) != 1:
        p.error('--llm_path requires exactly one --models selection')
    if any(h < 1 for h in args.pred_lens):
        p.error('pred_lens must be positive')
    if not args.run_tag or any(c in args.run_tag for c in '/\\:') or args.run_tag in ('.', '..'):
        p.error('run_tag must be a single directory name')
    plan = list(commands(args))
    for model, dataset, horizon, mode, command in plan:
        print(f'[{model} / {dataset} / H={horizon} / {mode}]', flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=str(ROOT), check=True)
    print(f'{len(plan)} commands {"planned" if args.dry_run else "completed"}.', flush=True)


if __name__ == '__main__':
    main()
