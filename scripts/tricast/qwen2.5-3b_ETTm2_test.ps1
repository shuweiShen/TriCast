param(
    [string]$Python = 'D:\Anaconda\envs\TimeSeries\python.exe',
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
$runArgs = @($driver, '--mode', 'test', '--models', 'qwen2.5-3b', '--datasets', 'ETTm2', '--run_tag', $RunTag, '--pred_lens')
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
