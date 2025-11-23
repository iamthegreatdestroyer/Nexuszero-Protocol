<#
setup-python-env.ps1
Minimal script to create and prepare a Python virtual environment and install requirements.
#>
param(
    [string]$EnvName = ".venv"
)

if (-not (Test-Path $EnvName)) {
    Write-Host "Creating virtual environment: $EnvName" -ForegroundColor Yellow
    python -m venv $EnvName
}

& "$EnvName\Scripts\Activate.ps1"

if (Test-Path "requirements.txt") {
    Write-Host "Installing Python dependencies from requirements.txt" -ForegroundColor Yellow
    pip install -r requirements.txt
} elseif (Test-Path "pyproject.toml") {
    Write-Host "pyproject.toml found. Please use a tool like poetry or pipx to install dependencies." -ForegroundColor Yellow
} else {
    Write-Host "No requirements.txt or pyproject.toml found. No packages installed." -ForegroundColor Gray
}

Write-Host "Python environment ready" -ForegroundColor Green
