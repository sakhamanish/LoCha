# Builds LoCha.exe with PyInstaller. Run in PowerShell from the LoCha folder:
#   .\build_exe.ps1
# Result: dist\LoCha\LoCha.exe (keep the whole dist\LoCha folder together).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No .venv found. Create it and install requirements first (see README)."
}

& $python -m pip install "pyinstaller==6.22.3"
if ($LASTEXITCODE -ne 0) { exit 1 }

& $python -m PyInstaller LoCha.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Done: dist\LoCha\LoCha.exe"
