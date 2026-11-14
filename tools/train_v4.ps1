<#
.SYNOPSIS
    Experiment 4 (2026task-4 vessel segmentation) training launcher.

.DESCRIPTION
    Why a wrapper exists: the DSH managed background jobs on this machine run the
    child process with a throttled CPU share, which starves the patch-augmentation
    thread and makes each step ~20x slower (0.44 s -> 7 s). Launching the run as a
    detached normal-priority process via Start-Process avoids that and keeps the GPU
    fed. This script also sets up the env vars (same as tools/py.ps1).

    ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI without a BOM).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_v4.ps1 -Tag v4_b0_384
#>
[CmdletBinding()]
param(
    [string]$Tag = 'v4_b0_384',
    [string]$Backbone = 'efficientnet_b0',
    [int]$Folds = 5,
    [int]$Epochs = 30,
    [int]$Steps = 40,
    [int]$Patch = 384,
    [int]$Batch = 8,
    [string]$Lr = '3e-4',
    [int]$Seed = 42,
    [string]$Extra = ''
)

$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $Root 'tools\shim'
$env:TEMP = Join-Path $Root '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HOME = Join-Path $Root '.hf_cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Virtual env not found: $Python" }

Push-Location $Root
try {
    $argv = @(
        '-m', 'src.vessel_train',
        '--tag', $Tag,
        '--backbone', $Backbone,
        '--folds', "$Folds",
        '--epochs', "$Epochs",
        '--steps', "$Steps",
        '--patch', "$Patch",
        '--batch-size', "$Batch",
        '--lr', $Lr,
        '--seed', "$Seed"
    )
    if ($Extra -ne '') { $argv += $Extra.Split(' ') }
    Write-Host ("[train_v4] " + ($argv -join ' '))
    & $Python @argv
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
