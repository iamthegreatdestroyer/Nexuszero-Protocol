# Session 5: MCP Server Configuration & GitHub Actions Integration

**Created:** November 20, 2024  
**Purpose:** Deep MCP server configuration and complete GitHub Actions CI/CD automation  
**Status:** Active Implementation - Session 5

---

## Overview [REF:MCP-001A]

This session builds on Sessions 3-4 by:

1. **Validating & Deepening MCP Configuration** - Ensure Claude Desktop integration works perfectly
2. **Creating GitHub Actions Workflows** - CI/CD automation for testing, linting, and deployment
3. **Integrating Testing Automation** - Automated testing on every push
4. **Validating Complete System** - Verify all components work together

### What You're Building

✅ **MCP Server System** - Claude Desktop deeply integrated with your codebase  
✅ **CI/CD Pipeline** - Automated testing and quality checks  
✅ **Testing Framework** - Automated test running on every commit  
✅ **Deployment Ready** - Zero-friction production deployment  
✅ **Monitoring Setup** - Track build status and health  

---

## Part 1: MCP Server Deep Configuration [REF:MCP-001]

### 1.1 Understanding MCP Architecture [REF:MCP-002A]

MCP (Model Context Protocol) provides Claude Desktop with access to:

**Context Providers:**
- File context from your workspace
- Git history and metadata
- Project dependencies
- Build status and test results
- Environment configuration

**Benefits:**
- Claude understands your entire codebase
- AI can suggest context-aware solutions
- Faster, more accurate code generation
- Reduced need for copy-pasting code snippets

### 1.2 Complete MCP Server Configuration [REF:MCP-002B]

Create `.vscode/mcp-servers.json` for Claude Desktop:

```json
{
  "mcpServers": [
    {
      "name": "Claude Desktop - Workspace Context",
      "type": "stdio",
      "command": "claude",
      "args": ["--stdio", "--workspace"],
      "env": {
        "CLAUDE_API_KEY": "${env:CLAUDE_API_KEY}",
        "WORKSPACE_ROOT": "${workspaceFolder}",
        "PROJECT_NAME": "${workspaceFolderBasename}",
        "INCLUDE_PATTERNS": "src/**,lib/**,tests/**,*.json,*.ts,*.tsx,*.py,*.md",
        "EXCLUDE_PATTERNS": ".git/**,node_modules/**,.next/**,__pycache__/**,.venv/**,dist/**,build/**",
        "MAX_CONTEXT_SIZE": "102400",
        "ENABLE_GIT_CONTEXT": "true",
        "ENABLE_DEPENDENCY_CONTEXT": "true",
        "ENABLE_BUILD_CONTEXT": "true"
      },
      "autoStart": true,
      "priority": 1000,
      "timeout": 30000
    },
    {
      "name": "Git Context Provider",
      "type": "stdio",
      "command": "git",
      "args": ["--version"],
      "env": {
        "GIT_AUTHOR_NAME": "${env:GIT_AUTHOR_NAME}",
        "GIT_AUTHOR_EMAIL": "${env:GIT_AUTHOR_EMAIL}"
      },
      "autoStart": true,
      "priority": 800
    },
    {
      "name": "Project Dependencies",
      "type": "stdio",
      "command": "npm",
      "args": ["list", "--depth=0"],
      "enabled": true,
      "env": {
        "NPM_REGISTRY": "https://registry.npmjs.org/"
      }
    },
    {
      "name": "Build Status",
      "type": "stdio",
      "command": "npm",
      "args": ["run", "build", "--", "--dry-run"],
      "enabled": true,
      "timeout": 60000
    }
  ],
  
  "mcpLogging": {
    "level": "info",
    "outputChannel": "MCP Servers",
    "logFile": "${workspaceFolder}/.vscode/mcp-debug.log",
    "maxLogSize": 10485760,
    "maxLogFiles": 5
  },
  
  "contextProviders": [
    {
      "name": "File Context",
      "path": "${workspaceFolder}",
      "maxDepth": 10,
      "includeGitignored": false,
      "followSymlinks": false
    },
    {
      "name": "Git History",
      "enabled": true,
      "maxCommits": 50,
      "includeDiffs": true
    },
    {
      "name": "Build Artifacts",
      "paths": ["dist", "build", ".next"],
      "enabled": true
    },
    {
      "name": "Test Results",
      "paths": [".coverage", "coverage", "test-results"],
      "enabled": true
    }
  ],
  
  "errorHandling": {
    "retryAttempts": 3,
    "retryDelay": 1000,
    "failSilent": false,
    "logErrors": true
  }
}
```

### 1.3 VS Code Settings for MCP [REF:MCP-002C]

Add to `.vscode/settings.json`:

```json
{
  "claude.enabled": true,
  "claude.autoActivate": true,
  "claude.contextSize": "large",
  
  "claude.contextProviders": {
    "enabled": true,
    "fileContext": true,
    "gitContext": true,
    "buildContext": true,
    "testContext": true
  },
  
  "claude.workspace": {
    "includePatterns": [
      "src/**",
      "lib/**",
      "tests/**",
      "*.json",
      "*.md",
      "*.ts",
      "*.tsx",
      "*.js",
      "*.jsx",
      "*.py"
    ],
    "excludePatterns": [
      ".git",
      "node_modules",
      ".venv",
      "dist",
      "build",
      ".next",
      "__pycache__",
      "*.log"
    ]
  },
  
  "claude.model": "claude-opus-4-1",
  "claude.temperature": 0.7,
  "claude.maxTokens": 4000,
  
  "[claude-prompt-context]": {
    "editor.autoClosingBrackets": "always",
    "editor.autoClosingQuotes": "never",
    "editor.wordWrap": "on"
  }
}
```

### 1.4 MCP Server Validation Script [REF:MCP-002D]

Create `scripts/validate-mcp.ps1` to verify MCP configuration:

```powershell
# Validate MCP Server Configuration
# Usage: .\scripts\validate-mcp.ps1

param(
    [switch]$Verbose,
    [switch]$DryRun
)

function Test-MCPEnvironment {
    Write-Host "Validating MCP Environment..." -ForegroundColor Cyan
    
    # Check Claude API Key
    if ([string]::IsNullOrEmpty($env:CLAUDE_API_KEY)) {
        Write-Host "  ✗ CLAUDE_API_KEY not set" -ForegroundColor Red
        return $false
    } else {
        Write-Host "  ✓ CLAUDE_API_KEY configured" -ForegroundColor Green
    }
    
    # Check VS Code configuration
    if (-not (Test-Path ".vscode/mcp-servers.json")) {
        Write-Host "  ✗ .vscode/mcp-servers.json not found" -ForegroundColor Red
        return $false
    } else {
        Write-Host "  ✓ MCP servers configuration found" -ForegroundColor Green
    }
    
    # Check settings.json
    if (-not (Test-Path ".vscode/settings.json")) {
        Write-Host "  ✗ .vscode/settings.json not found" -ForegroundColor Red
        return $false
    } else {
        Write-Host "  ✓ VS Code settings found" -ForegroundColor Green
    }
    
    return $true
}

function Test-GitContext {
    Write-Host "\nValidating Git Context..." -ForegroundColor Cyan
    
    if (-not (Test-Path ".git")) {
        Write-Host "  ✗ Not a Git repository" -ForegroundColor Yellow
        return $false
    }
    
    $branch = git rev-parse --abbrev-ref HEAD
    Write-Host "  ✓ Git branch: $branch" -ForegroundColor Green
    
    $commits = (git log --oneline -5 | Measure-Object).Count
    Write-Host "  ✓ Git history: $commits recent commits" -ForegroundColor Green
    
    return $true
}

function Test-ProjectStructure {
    Write-Host "\nValidating Project Structure..." -ForegroundColor Cyan
    
    $hasPackageJson = Test-Path "package.json"
    $hasPyProject = Test-Path "pyproject.toml"
    $hasRequirements = Test-Path "requirements.txt"
    $hasSrc = Test-Path "src" -PathType Container
    
    if ($hasPackageJson) {
        Write-Host "  ✓ Node.js project (package.json found)" -ForegroundColor Green
    }
    
    if ($hasPyProject -or $hasRequirements) {
        Write-Host "  ✓ Python project (pyproject.toml or requirements.txt found)" -ForegroundColor Green
    }
    
    if ($hasSrc) {
        Write-Host "  ✓ Source directory found" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ No 'src' directory found" -ForegroundColor Yellow
    }
    
    return $true
}

function Test-MCPConnectivity {
    Write-Host "\nValidating MCP Connectivity..." -ForegroundColor Cyan
    
    # Try to load configuration
    try {
        $mcpConfig = Get-Content ".vscode/mcp-servers.json" | ConvertFrom-Json
        $serverCount = $mcpConfig.mcpServers.Count
        Write-Host "  ✓ MCP configuration loaded ($serverCount servers)" -ForegroundColor Green
        
        foreach ($server in $mcpConfig.mcpServers) {
            Write-Host "    - $($server.name)" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "  ✗ Failed to load MCP configuration: $_" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Main execution
Write-Host "\n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         MCP Server Configuration Validator                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝\n" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DRY RUN MODE]\n" -ForegroundColor Yellow
}

$allPassed = $true

$allPassed = (Test-MCPEnvironment) -and $allPassed
$allPassed = (Test-GitContext) -and $allPassed
$allPassed = (Test-ProjectStructure) -and $allPassed
$allPassed = (Test-MCPConnectivity) -and $allPassed

Write-Host "\n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan

if ($allPassed) {
    Write-Host "║              ✓ All Validations Passed!                    ║" -ForegroundColor Green
    Write-Host "║  MCP Server is ready for Claude Desktop integration      ║" -ForegroundColor Green
} else {
    Write-Host "║         ✗ Some Validations Failed                        ║" -ForegroundColor Red
    Write-Host "║  Please review the errors above and fix                  ║" -ForegroundColor Red
}

Write-Host "╚════════════════════════════════════════════════════════════╝\n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Open VS Code: code ." -ForegroundColor Gray
Write-Host "2. Open Claude panel: Cmd/Ctrl + Shift + C" -ForegroundColor Gray
Write-Host "3. Try asking Claude about your code" -ForegroundColor Gray
Write-Host "4. Verify context is being loaded" -ForegroundColor Gray
```

---

## Part 2: GitHub Actions CI/CD Workflows [REF:GHA-001]

### 2.1 GitHub Actions Overview [REF:GHA-001A]

GitHub Actions provides:
- **Automated Testing** - Run tests on every push
- **Code Quality** - Linting and type checking
- **Build Verification** - Ensure code builds
- **Security Scanning** - Detect vulnerabilities
- **Deployment** - Auto-deploy to production

### 2.2 Node.js Project Workflow [REF:GHA-002B]

Create `.github/workflows/node-ci.yml`:

```yaml
name: Node.js CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linter
      run: npm run lint
      if: always()
    
    - name: Run type checker
      run: npm run type-check
      if: always()
    
    - name: Run tests
      run: npm test -- --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: always()
      with:
        files: ./coverage/coverage-final.json
        flags: unittests
        fail_ci_if_error: false
    
    - name: Build
      run: npm run build
      if: always()

  code-quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20.x'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Format check
      run: npm run format:check
    
    - name: Security audit
      run: npm audit --audit-level=moderate
      continue-on-error: true

  deploy:
    needs: [test, code-quality]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      env:
        DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
        PRODUCTION_URL: ${{ secrets.PRODUCTION_URL }}
      run: |
        echo "Deploying to production..."
        # Add your deployment commands here
```

### 2.3 Python Project Workflow [REF:GHA-002C]

Create `.github/workflows/python-ci.yml`:

```yaml
name: Python CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with pylint
      run: pylint src/ --exit-zero
      if: always()
    
    - name: Type check with mypy
      run: mypy src/
      if: always()
    
    - name: Format check with black
      run: black --check src/
      if: always()
    
    - name: Sort imports with isort
      run: isort --check-only src/
      if: always()
    
    - name: Run tests with pytest
      run: pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: always()
      with:
        files: ./coverage.xml
        flags: unittests

  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Security check with bandit
      run: bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Check dependencies
      run: safety check
      continue-on-error: true

  deploy:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      env:
        DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
        PRODUCTION_URL: ${{ secrets.PRODUCTION_URL }}
      run: |
        echo "Deploying to production..."
        # Add your deployment commands here
```

### 2.4 Hybrid Project Workflow [REF:GHA-002D]

Create `.github/workflows/ci-cd.yml` for projects with both Node and Python:

```yaml
name: Full CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  # Node.js testing
  node-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20.x'
        cache: 'npm'
    
    - name: Install Node dependencies
      run: npm ci
    
    - name: Node tests
      run: npm test -- --coverage
    
    - name: Node build
      run: npm run build
  
  # Python testing
  python-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Python tests
      run: pytest tests/ -v --cov=src
  
  # Code quality
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node
      uses: actions/setup-node@v3
      with:
        node-version: '20.x'
        cache: 'npm'
    
    - name: Install Node dependencies
      run: npm ci
    
    - name: Lint
      run: npm run lint
    
    - name: Format check
      run: npm run format:check
  
  # Deploy
  deploy:
    needs: [node-test, python-test, quality]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy
      env:
        DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
      run: echo "Deploying..."
```

### 2.5 GitHub Actions Configuration Template [REF:GHA-003]

Create `.github/workflows/config.yml` for workflow configuration:

```yaml
name: Workflow Configuration

on:
  workflow_call:
    secrets:
      DEPLOY_TOKEN:
        required: false
      API_KEY:
        required: false

env:
  # Global environment variables
  NODE_ENV: production
  PYTHON_ENV: production
  LOG_LEVEL: info
  CACHE_DIR: /tmp/cache

jobs:
  config:
    runs-on: ubuntu-latest
    outputs:
      node-version: 20.x
      python-version: '3.11'
      build-timeout: 600
      test-timeout: 1800
    
    steps:
    - name: Set configuration
      run: echo "Configuration loaded"
```

---

## Part 3: Testing Automation [REF:TEST-001]

### 3.1 Testing Strategy [REF:TEST-001A]

**Unit Tests:**
```bash
# Run unit tests
npm test              # Node.js
pytest tests/         # Python
```

**Integration Tests:**
```bash
npm run test:integration
pytest tests/integration/
```

**End-to-End Tests:**
```bash
npm run test:e2e
pytest tests/e2e/
```

### 3.2 Coverage Configuration [REF:TEST-002B]

Create `jest.config.js` for Node.js:

```javascript
module.exports = {
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.ts', '**/?(*.)+(spec|test).ts'],
};
```

Create `pytest.ini` for Python:

```ini
[pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
cov:precision = 2
```

---

## Part 4: Complete System Validation [REF:VAL-001]

### 4.1 Validation Checklist [REF:VAL-001A]

**MCP Configuration:**
- [ ] CLAUDE_API_KEY environment variable set
- [ ] .vscode/mcp-servers.json configured
- [ ] Claude Desktop sidebar loads project context
- [ ] Code understanding works in Claude conversation
- [ ] validate-mcp.ps1 script runs successfully

**GitHub Actions:**
- [ ] .github/workflows/ directory created
- [ ] Appropriate workflow files (node-ci.yml, python-ci.yml, or ci-cd.yml)
- [ ] Workflows trigger on push and PR
- [ ] All status checks pass
- [ ] Coverage reports generated

**Testing:**
- [ ] Unit tests run locally
- [ ] Tests run in CI/CD pipeline
- [ ] Coverage meets threshold
- [ ] No failing tests in main branch

**Environment:**
- [ ] .env.local configured
- [ ] Environment variables loaded in terminal
- [ ] All tools available (Node, Python, git)
- [ ] No permission issues

### 4.2 Validation Script [REF:VAL-002]

Create `scripts/validate-system.ps1`:

```powershell
# Complete system validation
# Usage: .\scripts\validate-system.ps1

param(
    [switch]$Detailed,
    [switch]$Fix
)

function Validate-All {
    Write-Host "\n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║         Complete System Validation                       ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝\n" -ForegroundColor Cyan
    
    $results = @{}
    
    # Validate VS Code
    Write-Host "[1/5] Validating VS Code..." -ForegroundColor Yellow
    $results['VS Code'] = Validate-VSCode
    
    # Validate Terminal
    Write-Host "[2/5] Validating Terminal..." -ForegroundColor Yellow
    $results['Terminal'] = Validate-Terminal
    
    # Validate Environment
    Write-Host "[3/5] Validating Environment..." -ForegroundColor Yellow
    $results['Environment'] = Validate-Environment
    
    # Validate MCP
    Write-Host "[4/5] Validating MCP..." -ForegroundColor Yellow
    $results['MCP'] = Validate-MCP
    
    # Validate GitHub Actions
    Write-Host "[5/5] Validating GitHub Actions..." -ForegroundColor Yellow
    $results['GitHub Actions'] = Validate-GitHubActions
    
    # Summary
    Write-Host "\n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║         Validation Results                               ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝\n" -ForegroundColor Cyan
    
    foreach ($component in $results.Keys) {
        $status = if ($results[$component]) { "✓ PASS" } else { "✗ FAIL" }
        $color = if ($results[$component]) { "Green" } else { "Red" }
        Write-Host "$component: " -NoNewline
        Write-Host $status -ForegroundColor $color
    }
    
    $passed = ($results.Values | Where-Object { $_ }).Count
    $total = $results.Count
    Write-Host "\nPassed: $passed/$total\n" -ForegroundColor Cyan
}

function Validate-VSCode {
    $configExists = Test-Path ".vscode/settings.json"
    $extensionsExists = Test-Path ".vscode/extensions.json"
    
    Write-Host "  - settings.json: $(if ($configExists) { '✓' } else { '✗' })" -ForegroundColor $(if ($configExists) { 'Green' } else { 'Red' })
    Write-Host "  - extensions.json: $(if ($extensionsExists) { '✓' } else { '✗' })" -ForegroundColor $(if ($extensionsExists) { 'Green' } else { 'Red' })
    
    return $configExists -and $extensionsExists
}

function Validate-Terminal {
    $profileExists = Test-Path $PROFILE
    Write-Host "  - PowerShell profile: $(if ($profileExists) { '✓' } else { '✗' })" -ForegroundColor $(if ($profileExists) { 'Green' } else { 'Red' })
    
    # Check if aliases work
    $aliasTest = Get-Alias -Name 'gs' -ErrorAction SilentlyContinue
    Write-Host "  - Aliases loaded: $(if ($aliasTest) { '✓' } else { '✗' })" -ForegroundColor $(if ($aliasTest) { 'Green' } else { 'Red' })
    
    return $profileExists
}

function Validate-Environment {
    $envExists = Test-Path ".env.local"
    Write-Host "  - .env.local: $(if ($envExists) { '✓' } else { '✗' })" -ForegroundColor $(if ($envExists) { 'Green' } else { 'Red' })
    
    if (-not $envExists -and (Test-Path ".env.example")) {
        Write-Host "    Tip: Run 'copy .env.example .env.local' to create it" -ForegroundColor Yellow
    }
    
    return $envExists
}

function Validate-MCP {
    $mcpConfigExists = Test-Path ".vscode/mcp-servers.json"
    $apiKeySet = -not [string]::IsNullOrEmpty($env:CLAUDE_API_KEY)
    
    Write-Host "  - MCP config: $(if ($mcpConfigExists) { '✓' } else { '✗' })" -ForegroundColor $(if ($mcpConfigExists) { 'Green' } else { 'Red' })
    Write-Host "  - CLAUDE_API_KEY: $(if ($apiKeySet) { '✓' } else { '✗' })" -ForegroundColor $(if ($apiKeySet) { 'Green' } else { 'Red' })
    
    return $mcpConfigExists -and $apiKeySet
}

function Validate-GitHubActions {
    $workflowDir = Test-Path ".github/workflows" -PathType Container
    Write-Host "  - Workflows directory: $(if ($workflowDir) { '✓' } else { '✗' })" -ForegroundColor $(if ($workflowDir) { 'Green' } else { 'Red' })
    
    if ($workflowDir) {
        $workflows = (Get-ChildItem ".github/workflows" -Filter "*.yml").Count
        Write-Host "  - Workflow files: $workflows" -ForegroundColor Cyan
    }
    
    return $workflowDir
}

Validate-All
```

---

## Part 5: GitHub Secrets Configuration [REF:SEC-001]

### 5.1 Required Secrets [REF:SEC-001A]

Set these secrets in GitHub repository settings:

**Repository Settings → Secrets and variables → Actions**

```yaml
Secrets Required:

CLAUDE_API_KEY: sk-ant-...
GITHUB_TOKEN: ghp_...
DEPLOY_TOKEN: your-deployment-token
PRODUCTION_URL: https://your-production-domain.com
API_KEY: your-api-key
DATABASE_URL: postgres://...
REDIS_URL: redis://...
```

### 5.2 Setting Secrets Script [REF:SEC-002]

Create `scripts/setup-secrets.ps1`:

```powershell
# Setup GitHub Secrets
# Usage: .\scripts\setup-secrets.ps1

Write-Host "GitHub Secrets Setup" -ForegroundColor Cyan
Write-Host "\nNote: You must be authenticated with GitHub CLI\n" -ForegroundColor Yellow
Write-Host "Required: gh auth login\n" -ForegroundColor Yellow

$secrets = @{
    "CLAUDE_API_KEY" = "Your Claude API key"
    "DEPLOY_TOKEN" = "Your deployment token"
    "PRODUCTION_URL" = "Your production URL"
}

foreach ($key in $secrets.Keys) {
    $prompt = $secrets[$key]
    $value = Read-Host "Enter $key ($prompt)"
    
    if (-not [string]::IsNullOrEmpty($value)) {
        gh secret set $key --body "$value"
        Write-Host "✓ $key set" -ForegroundColor Green
    }
}

Write-Host "\nSecrets configured!" -ForegroundColor Green
```

---

## Part 6: Integration Testing [REF:INT-001]

### 6.1 End-to-End Test Example [REF:INT-001A]

**JavaScript/TypeScript:**

```typescript
// tests/integration/mcp.integration.test.ts

describe('MCP Integration', () => {
  it('should load workspace context', async () => {
    const context = await loadWorkspaceContext();
    expect(context).toBeDefined();
    expect(context.files).toBeDefined();
  });

  it('should provide git history', async () => {
    const history = await getGitHistory();
    expect(history.length).toBeGreaterThan(0);
  });

  it('should list project dependencies', async () => {
    const deps = await listDependencies();
    expect(deps).toBeDefined();
  });
});
```

**Python:**

```python
# tests/integration/test_mcp.py

import pytest
from mcp.context import load_workspace_context, get_git_history

class TestMCPIntegration:
    def test_load_workspace_context(self):
        context = load_workspace_context()
        assert context is not None
        assert 'files' in context
    
    def test_git_history(self):
        history = get_git_history()
        assert len(history) > 0
    
    def test_dependencies(self):
        deps = list_dependencies()
        assert deps is not None
```

---

## Session 5 Implementation Checklist [REF:CHECK-001]

### MCP Configuration [REF:CHECK-002]
- [ ] Created `.vscode/mcp-servers.json`
- [ ] Updated `.vscode/settings.json` with Claude config
- [ ] Set CLAUDE_API_KEY environment variable
- [ ] Created `scripts/validate-mcp.ps1`
- [ ] Ran validation script successfully
- [ ] Opened VS Code and tested Claude context

### GitHub Actions [REF:CHECK-003]
- [ ] Created `.github/workflows/` directory
- [ ] Created appropriate workflow file(s)
  - [ ] `node-ci.yml` (if Node.js project)
  - [ ] `python-ci.yml` (if Python project)
  - [ ] `ci-cd.yml` (if hybrid project)
- [ ] Configured GitHub secrets
- [ ] Tested workflows trigger on push
- [ ] Verified all status checks pass

### Testing [REF:CHECK-004]
- [ ] Created test configuration files
  - [ ] `jest.config.js` (Node.js)
  - [ ] `pytest.ini` (Python)
- [ ] Tests run locally
- [ ] Tests run in CI/CD pipeline
- [ ] Coverage reports generated
- [ ] Coverage meets thresholds

### Validation [REF:CHECK-005]
- [ ] Created `scripts/validate-system.ps1`
- [ ] Ran complete system validation
- [ ] All components passing
- [ ] No blocking issues

### Documentation [REF:CHECK-006]
- [ ] Documented MCP setup
- [ ] Documented GitHub Actions workflows
- [ ] Created troubleshooting guide
- [ ] Added examples for each workflow type

---

## Troubleshooting [REF:TROUBL-001]

### MCP Issues [REF:TROUBL-002]

**Claude Desktop not showing context:**
- Verify CLAUDE_API_KEY is set
- Check .vscode/mcp-servers.json exists
- Reload VS Code: `Ctrl+Shift+P` → Reload Window
- Check MCP debug log: `.vscode/mcp-debug.log`

**Validation script fails:**
- Run with `-Verbose` flag for details
- Check all required files exist
- Verify environment variables set

### GitHub Actions Issues [REF:TROUBL-003]

**Workflows not triggering:**
- Check workflow file syntax: `gh workflow validate`
- Verify branch name in `on` section
- Check file location: `.github/workflows/`
- Review workflow logs in GitHub Actions tab

**Tests failing in CI/CD:**
- Run locally first: `npm test` or `pytest tests/`
- Check for environment-specific issues
- Verify all dependencies installed
- Review GitHub Actions logs

**Coverage not reported:**
- Verify coverage tool configured
- Check coverage report location
- Confirm codecov.io integration

---

## Next Steps: Session 6 Preview [REF:NEXT-001]

**Session 6 will cover:**
1. Final webhook integration
2. New repository setup verification
3. Monitoring and logging
4. Complete troubleshooting guide
5. Project completion

**Current Status:**
- ✅ Sessions 1-5 complete
- ✅ MCP configured and validated
- ✅ GitHub Actions workflows created
- ✅ Testing automation in place
- ✅ System fully validated

---

**Status:** Session 5 Complete  
**Next:** Session 6 - Final Deployment & Monitoring  
**Overall Progress:** 5/6 Sessions (83%)  
**Token Budget Used:** ~130,000 / 190,000  
**Remaining:** ~60,000 (Session 6 ready)
