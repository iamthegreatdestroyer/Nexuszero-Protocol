<#
init-project.ps1
A simple wrapper script to auto-detect project type and initialize Node/Python environments.
#>
param(
    [ValidateSet("node","python","both")]
    [string]$ProjectType = "both",
    [switch]$DryRun
)

function Ensure-Command {
    param([ool]$Condition, [string]$Message)
    if (-not $Condition) {
        Write-Host "✗ $Message" -ForegroundColor Red
        return $false
    }
    return $true
}

Write-Host "Initializing project environment..." -ForegroundColor Cyan

$hasPackage = Test-Path "package.json"
$hasRequirements = Test-Path "requirements.txt" -or Test-Path "pyproject.toml"

if ($ProjectType -eq "node" -or $ProjectType -eq "both") {
    if ($hasPackage) {
        Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
        if (-not $DryRun) { npm install }
    } else {
        Write-Host "No package.json found; skipping npm install" -ForegroundColor Gray
    }
}

if ($ProjectType -eq "python" -or $ProjectType -eq "both") {
    if ($hasRequirements -or Test-Path "pyproject.toml") {
        Write-Host "Setting up Python environment..." -ForegroundColor Yellow
        if (-not (Test-Path ".venv")) {
            Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
            if (-not $DryRun) {
                python -m venv .venv
            }
        }

        if (-not $DryRun) {
            & ".venv\Scripts\Activate.ps1"
            if (Test-Path "requirements.txt") {
                pip install -r requirements.txt
            } elseif (Test-Path "pyproject.toml") {
                Write-Host "pyproject.toml found. Consider using pipx/poetry to install dependencies." -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "No Python requirements found; skipping Python setup" -ForegroundColor Gray
    }
}

# Create .env.local from example if present
if (Test-Path ".env.example") {
    if (-not (Test-Path ".env.local")) {
        Copy-Item ".env.example" ".env.local"
        Write-Host "✓ Created .env.local from .env.example" -ForegroundColor Yellow
    } else {
        Write-Host ".env.local already exists; not overwriting" -ForegroundColor Gray
    }
}

Write-Host "✓ Project environment initialized" -ForegroundColor Green
