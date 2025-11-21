<#
setup-node-env.ps1
Minimal script to setup Node environment (npm install)
#>

if (Test-Path "package.json") {
    Write-Host "Installing npm dependencies from package.json" -ForegroundColor Yellow
    npm install
    Write-Host "✓ Node environment ready" -ForegroundColor Green
} else {
    Write-Host "No package.json found; skipping npm install" -ForegroundColor Gray
}
