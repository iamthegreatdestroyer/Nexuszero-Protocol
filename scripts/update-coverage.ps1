Param(
    [string]$WorkspacePath = "$PSScriptRoot/..",
    [int]$FailUnder = 90
)

$ErrorActionPreference = 'Stop'

Write-Host "[coverage] Running cargo tarpaulin..."
Push-Location "$WorkspacePath/nexuszero-crypto"

if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "cargo not found in PATH"
}

if (-not (Get-Command cargo-tarpaulin -ErrorAction SilentlyContinue)) {
    Write-Host "[coverage] Installing cargo-tarpaulin" -ForegroundColor Yellow
    cargo install cargo-tarpaulin | Out-Null
}

$raw = cargo tarpaulin --out Stdout 2>&1
Pop-Location

# Extract coverage percentage
$coverageLine = ($raw -split "`n") | Where-Object { $_ -match "coverage" -and $_ -match "%" } | Select-Object -Last 1
if (-not $coverageLine) { Write-Error "Unable to locate coverage line" }

if ($coverageLine -match "([0-9]+\.[0-9]+)%") {
    $coverage = [double]$Matches[1]
    Write-Host "[coverage] Detected: $coverage%" -ForegroundColor Green
} else {
    Write-Error "Failed to parse coverage percentage"
}

if ($coverage -lt $FailUnder) {
    Write-Error "Coverage $coverage% below threshold $FailUnder%"
}

$historyFile = Join-Path $WorkspacePath "docs/coverage/HISTORY.md"
if (-not (Test-Path $historyFile)) {
    Write-Error "History file not found at $historyFile"
}

$date = (Get-Date).ToString('yyyy-MM-dd')
# Determine delta from last entry
$history = Get-Content $historyFile
$lastEntry = $history | Where-Object { $_ -match "^\| [0-9]{4}-[0-9]{2}-[0-9]{2} \|" } | Select-Object -Last 1
$delta = "-"
if ($lastEntry -and $lastEntry -match "\| [0-9]{4}-[0-9]{2}-[0-9]{2} \| ([0-9]+\.[0-9]+)% \|") {
    $prev = [double]$Matches[1]
    $deltaValue = [Math]::Round($coverage - $prev, 2)
    if ($prev -ne $coverage) { $delta = "${deltaValue:+$deltaValue%}" } else { $delta = "+0.00%" }
}

$note = "Automated update"
$line = "| $date | $coverage% | $delta | $note |"

# Append if not already present for today
if (-not ($history -match "\| $date \|")) {
    Add-Content -Path $historyFile -Value $line
    Write-Host "[coverage] Appended history entry: $line"
} else {
    Write-Host "[coverage] Entry for $date already exists; skipping append" -ForegroundColor Yellow
}

Write-Host "[coverage] Done"
