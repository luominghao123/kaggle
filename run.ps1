<#
.SYNOPSIS
    Unified entry point: sets sandbox-compatible env vars, then calls the venv Python.

.DESCRIPTION
    Calling .venv\Scripts\python.exe directly fails for three reasons on this machine:
      1. The file sandbox denies writes into tempfile.mkdtemp() directories
         -> need PYTHONPATH to inject the shim.
      2. %TEMP% lives outside the workspace and is not writable
         -> point TEMP/TMP into .tmp.
      3. huggingface_hub defaults to ~/.cache/huggingface, also outside the workspace
         -> point HF_HOME into .hf_cache, and use the hf-mirror endpoint.
    This script handles all of them.

    NOTE: kept ASCII-only on purpose. Windows PowerShell 5.1 reads .ps1 as ANSI
    (GBK) unless the file carries a UTF-8 BOM, so non-ASCII text here breaks parsing.
    Chinese documentation lives in README.md and the Python sources.

.EXAMPLE
    .\run.ps1 -m src.data data
    .\run.ps1 -m src.train --data data --model convnext_tiny --img-size 384 --epochs 30
    .\run.ps1 -m src.predict --data data --runs runs/cnxt384 --out submissions/cnxt384.csv
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

$env:PYTHONPATH = Join-Path $Root 'tools\shim'
$env:TEMP = Join-Path $Root '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# timm pretrained weights: huggingface.co is unreachable here, use the mirror
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HOME = Join-Path $Root '.hf_cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Error "Virtual env not found: $Python -- see README.md section 1."
}

Push-Location $Root
try {
    & $Python @Rest
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
