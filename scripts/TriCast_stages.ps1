param(
    [string]$Python = 'D:\Anaconda\envs\TimeSeries\python.exe',
    [int]$Layers = 4
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    foreach ($stage in 0..5) {
        Write-Host "Checking TriCast stage $stage"
        & $Python run_tricast.py --mode smoke --tricast_stage $stage --llm_layers $Layers --batch_size 2 --output_dir logs/tricast_stages
        if ($LASTEXITCODE -ne 0) { throw "Stage $stage failed; stopping here." }
    }
} finally {
    Pop-Location
}
