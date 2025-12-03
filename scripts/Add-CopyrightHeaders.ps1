# Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
#
# This file is part of NexusZero Protocol.
#
# NexusZero Protocol is dual-licensed:
# 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
# 2. Commercial license: Contact legal@nexuszero.io

<#
.SYNOPSIS
    Adds copyright headers to all source files in the NexusZero Protocol project.

.DESCRIPTION
    This script scans the project directory and adds appropriate copyright headers
    to source files based on their file extension. It respects existing headers and
    will skip files that already have copyright notices.

.PARAMETER Path
    The root path of the project. Defaults to the script's parent directory.

.PARAMETER DryRun
    If specified, shows what would be done without making changes.

.PARAMETER Force
    If specified, replaces existing copyright headers with new ones.

.EXAMPLE
    .\Add-CopyrightHeaders.ps1 -DryRun
    Shows what files would be modified without making changes.

.EXAMPLE
    .\Add-CopyrightHeaders.ps1
    Adds copyright headers to all eligible files.
#>

param(
    [string]$Path = (Split-Path -Parent $PSScriptRoot),
    [switch]$DryRun,
    [switch]$Force,
    [switch]$Verbose
)

# Copyright header templates
$Headers = @{
    rust = @"
// Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
//
// This file is part of NexusZero Protocol.
//
// NexusZero Protocol is dual-licensed:
// 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
// 2. Commercial license: Contact legal@nexuszero.io
//
// Patent Pending - Contains proprietary innovations. See PATENT_CLAIMS.md.
//
// SECURITY WARNING: This software has not been independently audited.
// DO NOT use in production without proper security review.

"@
    
    python = @"
# Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
#
# This file is part of NexusZero Protocol.
#
# NexusZero Protocol is dual-licensed:
# 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
# 2. Commercial license: Contact legal@nexuszero.io
#
# Patent Pending - Contains proprietary innovations. See PATENT_CLAIMS.md.

"@

    typescript = @"
/**
 * Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
 *
 * This file is part of NexusZero Protocol.
 *
 * NexusZero Protocol is dual-licensed:
 * 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
 * 2. Commercial license: Contact legal@nexuszero.io
 *
 * Patent Pending - Contains proprietary innovations. See PATENT_CLAIMS.md.
 */

"@

    solidity = @"
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
//
// This file is part of NexusZero Protocol.
//
// NexusZero Protocol is dual-licensed:
// 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
// 2. Commercial license: Contact legal@nexuszero.io
//
// Patent Pending - Contains proprietary innovations.

"@

    shell = @"
# Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
#
# This file is part of NexusZero Protocol.
#
# NexusZero Protocol is dual-licensed:
# 1. AGPLv3 for open-source use: https://www.gnu.org/licenses/agpl-3.0.html
# 2. Commercial license: Contact legal@nexuszero.io

"@

    yaml = @"
# Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
# Licensed under AGPLv3. Commercial licenses: legal@nexuszero.io

"@

    dockerfile = @"
# Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
# Licensed under AGPLv3. Commercial licenses: legal@nexuszero.io
# Patent Pending.

"@

    css = @"
/**
 * Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
 * Licensed under AGPLv3. Commercial licenses: legal@nexuszero.io
 */

"@

    html = @"
<!--
  Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.
  Licensed under AGPLv3. Commercial licenses: legal@nexuszero.io
-->

"@
}

# File extension to header type mapping
$ExtensionMap = @{
    ".rs"     = "rust"
    ".py"     = "python"
    ".pyw"    = "python"
    ".ts"     = "typescript"
    ".tsx"    = "typescript"
    ".js"     = "typescript"
    ".jsx"    = "typescript"
    ".mjs"    = "typescript"
    ".sol"    = "solidity"
    ".sh"     = "shell"
    ".bash"   = "shell"
    ".ps1"    = "shell"
    ".psm1"   = "shell"
    ".yaml"   = "yaml"
    ".yml"    = "yaml"
    ".toml"   = "yaml"
    ".css"    = "css"
    ".scss"   = "css"
    ".less"   = "css"
    ".html"   = "html"
    ".htm"    = "html"
}

# Directories to exclude
$ExcludeDirs = @(
    "node_modules",
    "target",
    "dist",
    "build",
    ".git",
    ".venv",
    "venv",
    "venv-neural",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "checkpoints",
    "wandb",
    "logs"
)

# Files to exclude
$ExcludeFiles = @(
    "package-lock.json",
    "Cargo.lock",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.d.ts"
)

function Test-HasCopyright {
    param([string]$Content)
    return $Content -match "Copyright.*NexusZero|SPDX-License-Identifier"
}

function Test-ShouldSkip {
    param([string]$FilePath)
    
    $fileName = Split-Path -Leaf $FilePath
    $dirPath = Split-Path -Parent $FilePath
    
    # Check excluded directories
    foreach ($excludeDir in $ExcludeDirs) {
        if ($dirPath -match [regex]::Escape($excludeDir)) {
            return $true
        }
    }
    
    # Check excluded files
    foreach ($excludeFile in $ExcludeFiles) {
        if ($fileName -like $excludeFile) {
            return $true
        }
    }
    
    return $false
}

function Add-CopyrightHeader {
    param(
        [string]$FilePath,
        [string]$HeaderType,
        [switch]$DryRun,
        [switch]$Force
    )
    
    $header = $Headers[$HeaderType]
    if (-not $header) {
        Write-Warning "No header template for type: $HeaderType"
        return $false
    }
    
    $content = Get-Content -Path $FilePath -Raw -ErrorAction SilentlyContinue
    if (-not $content) {
        return $false
    }
    
    # Check if already has copyright
    if ((Test-HasCopyright -Content $content) -and (-not $Force)) {
        if ($Verbose) {
            Write-Host "  SKIP (has copyright): $FilePath" -ForegroundColor Yellow
        }
        return $false
    }
    
    # Handle shebang lines
    $shebang = ""
    if ($content -match "^(#![^\r\n]+[\r\n]+)") {
        $shebang = $Matches[1]
        $content = $content.Substring($shebang.Length)
    }
    
    # Handle existing SPDX for Solidity
    if ($HeaderType -eq "solidity" -and $content -match "^// SPDX-License-Identifier:") {
        # Remove existing SPDX line
        $content = $content -replace "^// SPDX-License-Identifier:[^\r\n]+[\r\n]+", ""
    }
    
    $newContent = $shebang + $header + $content
    
    if ($DryRun) {
        Write-Host "  WOULD ADD: $FilePath" -ForegroundColor Cyan
        return $true
    }
    
    Set-Content -Path $FilePath -Value $newContent -NoNewline
    Write-Host "  ADDED: $FilePath" -ForegroundColor Green
    return $true
}

# Main execution
Write-Host "==========================================================" -ForegroundColor Blue
Write-Host "  NexusZero Protocol - Copyright Header Tool" -ForegroundColor Blue
Write-Host "==========================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "Root Path: $Path"
Write-Host "Dry Run: $DryRun"
Write-Host "Force: $Force"
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

$stats = @{
    Scanned = 0
    Added = 0
    Skipped = 0
    Errors = 0
}

# Process all files
Get-ChildItem -Path $Path -Recurse -File | ForEach-Object {
    $file = $_
    $stats.Scanned++
    
    if (Test-ShouldSkip -FilePath $file.FullName) {
        $stats.Skipped++
        return
    }
    
    $ext = $file.Extension.ToLower()
    $headerType = $ExtensionMap[$ext]
    
    # Special case for Dockerfile
    if ($file.Name -eq "Dockerfile") {
        $headerType = "dockerfile"
    }
    
    if (-not $headerType) {
        return
    }
    
    try {
        if (Add-CopyrightHeader -FilePath $file.FullName -HeaderType $headerType -DryRun:$DryRun -Force:$Force) {
            $stats.Added++
        }
    }
    catch {
        Write-Warning "Error processing $($file.FullName): $_"
        $stats.Errors++
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Blue
Write-Host "  Summary" -ForegroundColor Blue
Write-Host "==========================================================" -ForegroundColor Blue
Write-Host "  Files Scanned:  $($stats.Scanned)"
Write-Host "  Headers Added:  $($stats.Added)" -ForegroundColor Green
Write-Host "  Files Skipped:  $($stats.Skipped)" -ForegroundColor Yellow
Write-Host "  Errors:         $($stats.Errors)" -ForegroundColor $(if ($stats.Errors -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($DryRun) {
    Write-Host "Run without -DryRun to apply changes." -ForegroundColor Yellow
}
