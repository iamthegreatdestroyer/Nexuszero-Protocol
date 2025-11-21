<#
  scripts/fix-readme-format.ps1
  Utility: Replace single-backtick code fences in README.md with proper triple-backtick fences.
  This is idempotent and safe; it does not change inline code.
#>

param(
    [string]$File = "README.md",
    [switch]$DryRun
)

Write-Host "Fixing code fence formatting in $File" -ForegroundColor Cyan

if (-not (Test-Path $File)) {
    Write-Error "$File not found. Run this from the repository root."
    exit 1
}

# Read all lines preserving EOL as array
$lines = Get-Content -LiteralPath $File -Encoding UTF8 -ErrorAction Stop

$convertedLines = @()
foreach ($line in $lines) {
    $trim = $line.TrimEnd()
    if ($trim -eq "`powershell") {
        $convertedLines += '```powershell'
        continue
    } elseif ($trim -eq "`bash") {
        $convertedLines += '```bash'
        continue
    } elseif ($trim -eq "`text") {
        $convertedLines += '```text'
        continue
    } elseif ($trim -eq "`python") {
        $convertedLines += '```python'
        continue
    } elseif ($trim -eq "`") {
        # solitary backtick -> closing fence
        $convertedLines += '```'
        continue
    } else {
        $convertedLines += $line
    }
}

if ($DryRun) {
    Write-Host "DryRun enabled - printing preview of modifications:" -ForegroundColor Yellow
    $convertedLines | Select-Object -First 100 | ForEach-Object { Write-Host $_ }
    exit 0
}

Write-Host "Writing formatted README to $File" -ForegroundColor Green
[System.IO.File]::WriteAllLines((Resolve-Path $File).Path, $convertedLines, [System.Text.Encoding]::UTF8)
Write-Host "README formatting fix complete." -ForegroundColor Green
