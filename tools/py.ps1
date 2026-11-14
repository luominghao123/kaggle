<#
.SYNOPSIS
    Same environment setup as run.ps1, but stderr-friendly (no ErrorActionPreference=Stop).

.DESCRIPTION
    run.ps1 sets $ErrorActionPreference = 'Stop', so any library that prints a harmless
    warning to stderr (timm does this while adapting pretrained weights) turns into a
    NativeCommandError that swallows the real stdout. This wrapper keeps the same env vars
    but lets stderr flow through unchanged, so warnings stay warning-shaped.

    ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI without a BOM).

.EXAMPLE
    .\tools\py.ps1 -m src.fovea_model effnet_b0
    .\tools\py.ps1 -m src.fovea_data --selftest
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

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
if (-not (Test-Path $Python)) {
    Write-Error "Virtual env not found: $Python"
}

Push-Location $Root
try {
    & $Python @Rest
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
