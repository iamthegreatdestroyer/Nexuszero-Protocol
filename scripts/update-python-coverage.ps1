Param(
    [string]$WorkspacePath = "$PSScriptRoot/..",
    [int]$FailUnder = 90
)

$ErrorActionPreference = 'Stop'

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
}

Write-Host "[pycov] Running pytest with coverage..."
Push-Location "$WorkspacePath/nexuszero-optimizer"

if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found in PATH"
}

# Create coverage XML
$env:PYTHONPATH = "$PWD/src"
$pytest = Get-Command pytest -ErrorAction SilentlyContinue
if (-not $pytest) { Write-Error "pytest not installed" }

pytest tests `
  -v `
  --maxfail=1 `
  --cov=nexuszero_optimizer `
  --cov-report=term-missing `
  --cov-report=xml:coverage.xml `
  --cov-report=html:htmlcov

if ($LASTEXITCODE -ne 0) { Write-Error "pytest failed" }

$covXmlPath = Join-Path $PWD 'coverage.xml'
if (-not (Test-Path $covXmlPath)) { Write-Error "coverage.xml not found at $covXmlPath" }

# Parse coverage percent from XML (line-rate attribute)
[xml]$xml = Get-Content $covXmlPath
$lineRate = $xml.coverage.'line-rate'
if (-not $lineRate) { Write-Error "Failed to parse line-rate from coverage.xml" }
$coverage = [Math]::Round([double]$lineRate * 100, 2)
Write-Host "[pycov] Detected Python coverage: $coverage%" -ForegroundColor Green

if ($coverage -lt $FailUnder) {
    Write-Error "Python coverage $coverage% below threshold $FailUnder%"
}

# Update HISTORY and badge under docs/coverage/python
$pythonCoverageDir = Join-Path $WorkspacePath 'docs/coverage/python'
New-DirectoryIfMissing $pythonCoverageDir

$historyFile = Join-Path $pythonCoverageDir 'HISTORY.md'
if (-not (Test-Path $historyFile)) {
    @(
        '# Python Coverage History'
        ''
        '| Date       | Coverage | Delta  | Notes         |'
        '| ---------- | -------- | ------ | ------------- |'
    ) | Set-Content -Path $historyFile -Encoding UTF8
}

$date = (Get-Date).ToString('yyyy-MM-dd')
$history = Get-Content $historyFile
$lastEntry = $history | Where-Object { $_ -match "^\| [0-9]{4}-[0-9]{2}-[0-9]{2} \|" } | Select-Object -Last 1
$delta = "-"
if ($lastEntry -and $lastEntry -match "\| [0-9]{4}-[0-9]{2}-[0-9]{2} \| ([0-9]+\.[0-9]+)% \|") {
    $prev = [double]$Matches[1]
    $diff = [Math]::Round($coverage - $prev, 2)
    if ($prev -ne $coverage) { $sign = if ($diff -ge 0) { "+" } else { "" }; $delta = "$sign$diff%" } else { $delta = "+0.00%" }
}

$line = "| $date | $coverage% | $delta | Automated update |"
if (-not ($history -match "\| $date \|")) {
    Add-Content -Path $historyFile -Value $line
    Write-Host "[pycov] Appended history entry: $line"
} else {
    Write-Host "[pycov] Entry for $date already exists; skipping append" -ForegroundColor Yellow
}

# Generate badge (PyCoverage)
$badgeFile = Join-Path $pythonCoverageDir 'badge.svg'
$color = if ($coverage -ge 90) { 'brightgreen' } elseif ($coverage -ge 80) { 'green' } elseif ($coverage -ge 70) { 'yellowgreen' } elseif ($coverage -ge 60) { 'yellow' } else { 'red' }
$escaped = [uri]::EscapeDataString("$coverage%")
$badgeUrl = "https://img.shields.io/badge/PyCoverage-$escaped-$color"
try {
    Invoke-WebRequest -Uri $badgeUrl -OutFile $badgeFile -UseBasicParsing
    Write-Host "[pycov] Saved Python badge to $badgeFile"
} catch {
    Write-Warning "[pycov] Failed to download badge: $_"
}

# Update README with PyCoverage badge
$readme = Join-Path $WorkspacePath 'README.md'
if (Test-Path $readme) {
    $content = Get-Content $readme -Raw
    $pattern = '!\[PyCoverage\]\(https://img\.shields\.io/badge/PyCoverage-[^)]+\)'
    $replacement = "![PyCoverage]($badgeUrl)"
    if ($content -match $pattern) {
        $new = $content -replace $pattern, $replacement
    } else {
        # Insert after existing Coverage badge if present, otherwise at top badges block
        $lines = $content -split "\r?\n"
        $idx = ($lines | Select-String -Pattern '!\[Coverage\]\(').LineNumber
        if ($idx) {
            $insertAt = [int]$idx
            $lines = @($lines[0..($insertAt-1)]) + @($replacement) + @($lines[$insertAt..($lines.Length-1)])
            $new = ($lines -join "`n")
        } else {
            $new = $replacement + "`n" + $content
        }
    }
    if ($new -ne $content) { Set-Content -Path $readme -Value $new -NoNewline }
}

Pop-Location
Write-Host "[pycov] Done"
