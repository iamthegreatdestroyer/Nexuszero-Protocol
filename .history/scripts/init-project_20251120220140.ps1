<#
init-project.ps1
A robust wrapper script to auto-detect project type and initialize Node/Python environments.
Supports DryRun, Force, NoNode, NoPython flags and creates logs/init-project.log
#>
param(
    [ValidateSet("node","python","both")]
    [string]$ProjectType = "both",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoNode,
    [switch]$NoPython
)

function Ensure-Command {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        Write-Host "✗ $Message" -ForegroundColor Red
        return $false
    }
    return $true
}

function Log {
    param([string]$msg, [string]$level = 'INFO')
    $timestamp = Get-Date -Format s
    $line = "[$timestamp] [$level] $msg"
    Write-Host $line
    try {
        $logDir = Join-Path -Path (Get-Location) -ChildPath "logs"
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        $logFile = Join-Path -Path $logDir -ChildPath "init-project.log"
        Add-Content -Path $logFile -Value $line -Encoding UTF8
    } catch { }
}

Log "Initializing project environment..."

$hasPackage = Test-Path "package.json"
$hasRequirements = (Test-Path "requirements.txt" -or Test-Path "pyproject.toml")

# If user opts out
if ($NoNode -and $NoPython) {
    Log "Both 'NoNode' and 'NoPython' flags are set. Nothing to do." 'WARN'
    return
}

if (($ProjectType -eq "node" -or $ProjectType -eq "both") -and -not $NoNode) {
    if ($hasPackage) {
        Log "Installing npm dependencies..."
        if (-not $DryRun) {
            if (Get-Command npm -ErrorAction SilentlyContinue) {
                # Prefer npm ci if lockfile exists
                if (Test-Path "package-lock.json") { npm ci } else { npm install }
            } else {
                Log "npm not found; skipping npm install" 'ERROR'
            }
        }
    } else {
        Log "No package.json found; skipping npm install" 'INFO'
    }
}

if (($ProjectType -eq "python" -or $ProjectType -eq "both") -and -not $NoPython) {
    if ($hasRequirements) {
        Log "Setting up Python environment..."
        if (-not (Test-Path ".venv")) {
            Log "Creating virtual environment (.venv)..."
            if (-not $DryRun) {
                if (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv } else {
                    Log "Python not found. Skipping python venv creation" 'ERROR'
                }
            }
        }

        if (-not $DryRun) {
            # Install dependencies in venv without relying on activate in non-interactive context
            $pip = Join-Path -Path ".venv" -ChildPath "Scripts\pip.exe"
            if (Test-Path $pip) {
                if (Test-Path "requirements.txt") { & $pip install -r requirements.txt }
                elseif (Test-Path "pyproject.toml") { Log "pyproject.toml found. Use poetry or pipx to install deps." 'INFO' }
            } else {
                Log "pip not found in venv; attempting global pip" 'WARN'
                if (Get-Command pip -ErrorAction SilentlyContinue) { pip install -r requirements.txt } else { Log "pip not found; skipping dependency install" 'ERROR' }
            }
        }
    } else {
        Log "No Python requirements found; skipping Python setup" 'INFO'
    }
}

# Create .env.local from example if present
if (Test-Path ".env.example") {
    if (-not (Test-Path ".env.local") -or $Force) {
        Log "Copying .env.example to .env.local (Force: $Force)"
        if (-not $DryRun) { Copy-Item ".env.example" ".env.local" -Force:$Force }
        Log "Created .env.local from .env.example"
    } else {
        Log ".env.local already exists; not overwriting (use -Force to overwrite)" 'INFO'
    }
}

Log "Project environment initialized" 'INFO'
<#
init-project.ps1
A robust wrapper script to auto-detect project type and initialize Node/Python environments.
Supports DryRun, Force, NoNode, NoPython flags and creates logs/init-project.log
#>
param(
    [ValidateSet("node","python","both")]
    [string]$ProjectType = "both",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoNode,
    [switch]$NoPython
)

function Ensure-Command {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        Write-Host "✗ $Message" -ForegroundColor Red
        return $false
    }
    return $true
}

function Log {
    param([string]$msg, [string]$level = 'INFO')
    $timestamp = Get-Date -Format s
    $line = "[$timestamp] [$level] $msg"
    Write-Host $line
    try {
        $logDir = Join-Path -Path (Get-Location) -ChildPath "logs"
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        $logFile = Join-Path -Path $logDir -ChildPath "init-project.log"
        Add-Content -Path $logFile -Value $line -Encoding UTF8
    } catch { }
}

Log "Initializing project environment..."

$hasPackage = Test-Path "package.json"
$hasRequirements = (Test-Path "requirements.txt" -or Test-Path "pyproject.toml")

# If user opts out
if ($NoNode -and $NoPython) {
    Log "Both 'NoNode' and 'NoPython' flags are set. Nothing to do." 'WARN'
    return
}

if (($ProjectType -eq "node" -or $ProjectType -eq "both") -and -not $NoNode) {
    if ($hasPackage) {
        Log "Installing npm dependencies..."
        if (-not $DryRun) {
            if (Get-Command npm -ErrorAction SilentlyContinue) {
                # Prefer npm ci if lockfile exists
                if (Test-Path "package-lock.json") { npm ci } else { npm install }
            } else {
                Log "npm not found; skipping npm install" 'ERROR'
            }
        }
    } else {
        Log "No package.json found; skipping npm install" 'INFO'
    }
}

if (($ProjectType -eq "python" -or $ProjectType -eq "both") -and -not $NoPython) {
    if ($hasRequirements) {
        Log "Setting up Python environment..."
        if (-not (Test-Path ".venv")) {
            Log "Creating virtual environment (.venv)..."
            if (-not $DryRun) {
                if (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv } else {
                    Log "Python not found. Skipping python venv creation" 'ERROR'
                }
            }
        }

        if (-not $DryRun) {
            # Install dependencies in venv without relying on activate in non-interactive context
            $pip = Join-Path -Path ".venv" -ChildPath "Scripts\pip.exe"
            if (Test-Path $pip) {
                if (Test-Path "requirements.txt") { & $pip install -r requirements.txt }
                elseif (Test-Path "pyproject.toml") { Log "pyproject.toml found. Use poetry or pipx to install deps." 'INFO' }
            } else {
                Log "pip not found in venv; attempting global pip" 'WARN'
                if (Get-Command pip -ErrorAction SilentlyContinue) { pip install -r requirements.txt } else { Log "pip not found; skipping dependency install" 'ERROR' }
            }
        }
    } else {
        Log "No Python requirements found; skipping Python setup" 'INFO'
    }
}

# Create .env.local from example if present
if (Test-Path ".env.example") {
    if (-not (Test-Path ".env.local") -or $Force) {
        Log "Copying .env.example to .env.local (Force: $Force)"
        if (-not $DryRun) { Copy-Item ".env.example" ".env.local" -Force:$Force }
        Log "Created .env.local from .env.example"
    } else {
        Log ".env.local already exists; not overwriting (use -Force to overwrite)" 'INFO'
    }
}

Log "Project environment initialized" 'INFO'
<#
init-project.ps1
A simple wrapper script to auto-detect project type and initialize Node/Python environments.
#>
param(
    [ValidateSet("node","python","both")]
    [string]$ProjectType = "both",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoNode,
    [switch]$NoPython
)

function Ensure-Command {
    param([ool]$Condition, [string]$Message)
    if (-not $Condition) {
        Write-Host "✗ $Message" -ForegroundColor Red
        return $false
    }
    return $true
}

function Log {
    param([string]$msg, [string]$level = 'INFO')
    $timestamp = Get-Date -Format s
    $line = "[$timestamp] [$level] $msg"
    Write-Host $line
    try {
        $logDir = Join-Path -Path (Get-Location) -ChildPath "logs"
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        $logFile = Join-Path -Path $logDir -ChildPath "init-project.log"
        Add-Content -Path $logFile -Value $line -Encoding UTF8
    } catch { }
}

Log "Initializing project environment..."

$hasPackage = Test-Path "package.json"
$hasRequirements = (Test-Path "requirements.txt" -or Test-Path "pyproject.toml")

# If user opts out
if ($NoNode -and $NoPython) {
    Log "Both 'NoNode' and 'NoPython' flags are set. Nothing to do." 'WARN'
    return
}

if (($ProjectType -eq "node" -or $ProjectType -eq "both") -and -not $NoNode) {
    if ($hasPackage) {
        Log "Installing npm dependencies..."
        if (-not $DryRun) {
            if (Get-Command npm -ErrorAction SilentlyContinue) {
                # Prefer npm ci if lockfile exists
                if (Test-Path "package-lock.json") { npm ci } else { npm install }
            } else {
                Log "npm not found; skipping npm install" 'ERROR'
            }
        }
    } else {
        Log "No package.json found; skipping npm install" 'INFO'
    }
}

if (($ProjectType -eq "python" -or $ProjectType -eq "both") -and -not $NoPython) {
    if ($hasRequirements -or Test-Path "pyproject.toml") {
        Log "Setting up Python environment..."
        if (-not (Test-Path ".venv")) {
            Log "Creating virtual environment (.venv)..."
            if (-not $DryRun) {
                if (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv } else {
                    Log "Python not found. Skipping python venv creation" 'ERROR'
                }
            }
        }

        if (-not $DryRun) {
            # Install dependencies in venv without relying on activate in non-interactive context
            $pip = Join-Path -Path ".venv" -ChildPath "Scripts\pip.exe"
            if (Test-Path $pip) {
                if (Test-Path "requirements.txt") { & $pip install -r requirements.txt } 
                elseif (Test-Path "pyproject.toml") { Log "pyproject.toml found. Use poetry or pipx to install deps." 'INFO' }
            } else {
                Log "pip not found in venv; attempting global pip" 'WARN'
                if (Get-Command pip -ErrorAction SilentlyContinue) { pip install -r requirements.txt } else { Log "pip not found; skipping dependency install" 'ERROR' }
            }
        }
    } else {
        Log "No Python requirements found; skipping Python setup" 'INFO'
    }
}

# Create .env.local from example if present
if (Test-Path ".env.example") {
    if (-not (Test-Path ".env.local") -or $Force) {
        Log "Copying .env.example to .env.local (Force: $Force)"
        if (-not $DryRun) { Copy-Item ".env.example" ".env.local" -Force:$Force }
        Log "Created .env.local from .env.example"
    } else {
        Log ".env.local already exists; not overwriting (use -Force to overwrite)" 'INFO'
    }
}

Log "Project environment initialized" 'INFO'
