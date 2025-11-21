Param(
    [string]$HistoryFile = "$PSScriptRoot/../docs/coverage/HISTORY.md",
    [string]$OutputFile = "$PSScriptRoot/../docs/coverage/badge.svg",
    [string]$ReadmeFile = "$PSScriptRoot/../README.md"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $HistoryFile)) {
    Write-Error "History file not found: $HistoryFile"
}

# Parse latest coverage from HISTORY.md
$history = Get-Content $HistoryFile
$lastEntry = $history | Where-Object { $_ -match "^\| [0-9]{4}-[0-9]{2}-[0-9]{2} \|" } | Select-Object -Last 1

if (-not $lastEntry) {
    Write-Error "No coverage entries found in $HistoryFile"
}

if ($lastEntry -match "\| [0-9]{4}-[0-9]{2}-[0-9]{2} \| ([0-9]+\.[0-9]+)% \|") {
    $coverage = [double]$Matches[1]
} else {
    Write-Error "Failed to parse coverage from entry: $lastEntry"
}

Write-Host "[badge] Latest coverage: $coverage%"

# Determine badge color
$color = if ($coverage -ge 90) { "brightgreen" } elseif ($coverage -ge 80) { "green" } elseif ($coverage -ge 70) { "yellowgreen" } elseif ($coverage -ge 60) { "yellow" } else { "red" }

# Generate SVG badge
$escapedCoverage = [uri]::EscapeDataString("$coverage%")
$badgeUrl = "https://img.shields.io/badge/Coverage-$escapedCoverage-$color"

Write-Host "[badge] Badge URL: $badgeUrl"

# Download badge SVG
try {
    Invoke-WebRequest -Uri $badgeUrl -OutFile $OutputFile -UseBasicParsing
    Write-Host "[badge] Saved badge to $OutputFile"
} catch {
    Write-Warning "Failed to download badge: $_"
}

# Update README.md shield
if (Test-Path $ReadmeFile) {
    $readme = Get-Content $ReadmeFile -Raw
    $pattern = '!\[Coverage\]\(https://img\.shields\.io/badge/Coverage-[^)]+\)'
    $replacement = "![Coverage]($badgeUrl)"
    
    if ($readme -match $pattern) {
        $newReadme = $readme -replace $pattern, $replacement
        if ($newReadme -ne $readme) {
            Set-Content -Path $ReadmeFile -Value $newReadme -NoNewline
            Write-Host "[badge] Updated README.md with new coverage badge"
        } else {
            Write-Host "[badge] README.md already has correct coverage badge"
        }
    } else {
        Write-Warning "Coverage badge pattern not found in README.md"
    }
} else {
    Write-Warning "README.md not found at $ReadmeFile"
}

Write-Host "[badge] Done"
