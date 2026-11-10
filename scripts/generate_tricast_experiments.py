"""Generate the 28 model/dataset configs and their Windows train/test wrappers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['gpt2-xl', 'gpt2-large', 'gpt2', 'qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'tinyllama-1.1b']


def main():
    configs = ROOT / 'configs' / 'experiments'
    scripts = ROOT / 'scripts' / 'tricast'
    configs.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            config = {
                'llm_name': model, 'llm_path': None, 'llm_layers': None,
                'llm_dtype': 'float32' if model == 'gpt2' else 'bfloat16',
                'tricast_stage': 5, 'seq_len': 672, 'pred_len': 96, 'patch_len': 96,
                'd_model': 128, 'd_ff': 128, 'n_heads': 8, 'num_experts': 4,
                'msdr_scales': [16, 32, 64], 'msdr_max_lag': 8, 'msdr_rho_floor': 0.1,
                'dropout': 0.1, 'data': dataset, 'root_path': './dataset/ETT-small',
                'data_path': dataset + '.csv', 'features': 'M', 'target': 'OT',
                'freq': 't' if dataset.startswith('ETTm') else 'h',
                'percent': 100, 'batch_size': 2 if model != 'gpt2' else 4,
                'num_workers': 0, 'train_epochs': 10, 'patience': 3,
                'learning_rate': 0.0001, 'weight_decay': 0.01, 'seed': 2021,
                'amp': False, 'max_batches': 0,
                'output_dir': f'./checkpoints/tricast_single/{model}/{dataset}'
            }
            (configs / f'{model}_{dataset}.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
            for mode in ['train', 'test']:
                content = '''param(
    [string]$Python = 'D:\\Anaconda\\envs\\TimeSeries\\python.exe',
    [int[]]$PredLens = @(96, 192, 336, 720),
    [string]$RunTag = 'default',
    [int]$Layers = 0,
    [int]$BatchSize = 0,
    [int]$Epochs = 0,
    [int]$MaxBatches = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$project = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$driver = Join-Path $project 'scripts/run_tricast_matrix.py'
$runArgs = @($driver, '--mode', 'MODE', '--models', 'MODEL', '--datasets', 'DATASET', '--run_tag', $RunTag, '--pred_lens')
$runArgs += @($PredLens | ForEach-Object { [string]$_ })
if ($Layers -gt 0) { $runArgs += @('--llm_layers', [string]$Layers) }
if ($BatchSize -gt 0) { $runArgs += @('--batch_size', [string]$BatchSize) }
if ($Epochs -gt 0) { $runArgs += @('--train_epochs', [string]$Epochs) }
if ($MaxBatches -gt 0) { $runArgs += @('--max_batches', [string]$MaxBatches) }
if ($DryRun) { $runArgs += '--dry_run' }
Push-Location $project
try {
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. See error above.' }
} finally { Pop-Location }
'''.replace('MODEL', model).replace('DATASET', dataset).replace('MODE', mode)
                (scripts / f'{model}_{dataset}_{mode}.ps1').write_text(content, encoding='utf-8')
    print('Generated 28 configs and 56 PowerShell scripts.')


if __name__ == '__main__':
    main()
