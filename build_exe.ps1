# Builds LoCha.exe with PyInstaller. Run in PowerShell from the LoCha folder:
#   .\build_exe.ps1
# Result: dist\LoCha\LoCha.exe (keep the whole dist\LoCha folder together).
#
# Success is judged by exit codes, not by $ErrorActionPreference = "Stop":
# PyInstaller writes its normal progress messages to stderr, which Windows
# PowerShell 5.1 would otherwise treat as an error and stop the build at the
# first "INFO" line.
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Create it and install requirements first (see README)." -ForegroundColor Red
    exit 1
}

& $python -m pip install "pyinstaller==6.22.3"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller failed (see messages above)." -ForegroundColor Red
    exit 1
}

& $python -m PyInstaller LoCha.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed (see messages above)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done: dist\LoCha\LoCha.exe" -ForegroundColor Green
