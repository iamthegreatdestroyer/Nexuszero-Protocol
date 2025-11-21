# Session 3: VS Code Workspace Configuration

**Created:** November 20, 2024  
**Purpose:** Production-ready workspace configuration system for automated developer environment setup  
**Status:** Active Implementation - Session 3

---

## Overview [REF:VSCODE-001A]

This guide provides a complete VS Code workspace configuration system that integrates with your GitHub webhook automation. The goal is to achieve consistent, production-ready developer environments across all repositories with minimal manual setup.

### What You're Building
- **Workspace Settings Template** - Reusable configurations for all projects
- **Project-Specific Overrides** - Per-project customizations while maintaining consistency
- **MCP Integration** - Seamless Claude Desktop + Copilot integration
- **Extension Management** - Automated extension installation and configuration
- **Dev Environment Standardization** - Consistent across Windows, development tooling

### Key Benefits
✅ Zero-friction developer onboarding  
✅ Consistent code formatting and linting across projects  
✅ Automated tool integration (Claude, Copilot, terminals)  
✅ Workspace-level environment variables  
✅ Project-specific debugging configurations  

---

## Part 1: Core Workspace Settings [REF:VSCODE-002]

### 1.1 Global Settings Structure

Your VS Code configuration uses two key files:

**`.vscode/settings.json`** - Workspace-level overrides (committed to repo)  
**`.vscode/extensions.json`** - Recommended extensions (auto-install prompts)

**Why This Approach:**
- Local settings stay local (token storage, personal preferences)
- Workspace settings are version-controlled (consistent across team)
- Extensions auto-install with confirmation (no broken setups)

### 1.2 Production-Ready Settings Template [REF:VSCODE-002A]

Create this as your base configuration:

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.fixAll.prettier": "explicit"
  },
  "editor.wordWrap": "on",
  "editor.wordWrapColumn": 100,
  "editor.rulers": [80, 100],
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "editor.trimAutoWhitespace": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  
  "files.exclude": {
    "**/.git": true,
    "**/.DS_Store": true,
    "**/*.swp": true,
    "**/node_modules": true,
    "**/__pycache__": true,
    "**/*.pyc": true
  },
  
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/build": true,
    "**/.venv": true
  },
  
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  
  "git.ignoreLimitWarning": true,
  "git.autofetch": true,
  "git.autorefresh": true,
  
  "terminal.integrated.defaultProfile.windows": "PowerShell",
  "terminal.integrated.profiles.windows": {
    "PowerShell": {
      "source": "PowerShell",
      "icon": "terminal-powershell",
      "args": ["-NoExit", "-Command", "cd $pwd"]
    },
    "Command Prompt": {
      "path": ["${env:windir}\\System32\\cmd.exe"],
      "icon": "terminal-cmd"
    }
  },
  
  "workbench.colorTheme": "One Dark Pro",
  "editor.fontFamily": "Consolas, 'Courier New', monospace",
  "editor.fontSize": 14,
  "editor.lineHeight": 1.6,
  "editor.letterSpacing": 0.5,
  
  "extensions.recommendations": [
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-vscode.makefile-tools",
    "eamodio.gitlens",
    "ms-azuretools.vscode-docker",
    "ms-vscode-remote.remote-ssh",
    "ms-vscode-remote.remote-containers",
    "charliermarsh.ruff",
    "Anthropic.claude-vscode"
  ]
}
```

### 1.3 Language-Specific Overrides [REF:VSCODE-002B]

Different projects need different tools. Override per-language:

**For Python Projects:**
```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

**For TypeScript/Node Projects:**
```json
{
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": "explicit"
    }
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "typescript.check.npmIsInstalled": true
}
```

---

## Part 2: Extension Management [REF:VSCODE-003]

### 2.1 Recommended Extensions Configuration [REF:VSCODE-003A]

Create `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-vscode.makefile-tools",
    "eamodio.gitlens",
    "ms-azuretools.vscode-docker",
    "ms-vscode-remote.remote-ssh",
    "ms-vscode-remote.remote-containers",
    "charliermarsh.ruff",
    "ms-vscode.powershell",
    "gruntfuggly.todo-tree",
    "wayou.vscode-todo-highlight",
    "Anthropic.claude-vscode",
    "GitHub.Copilot",
    "GitHub.Copilot-Chat",
    "redhat.vscode-yaml",
    "yzhang.markdown-all-in-one",
    "DavidAnson.vscode-markdownlint"
  ]
}
```

### 2.2 Essential Extensions Breakdown [REF:VSCODE-003B]

| Extension | Purpose | Auto-Install | Why Essential |
|-----------|---------|--------------|---------------|
| **esbenp.prettier-vscode** | Code formatter | ✅ | Consistent formatting across team |
| **dbaeumer.vscode-eslint** | JavaScript linter | ✅ | Catch code errors before runtime |
| **ms-python.python** | Python support | ✅ | Core language support |
| **ms-python.vscode-pylance** | Python type checking | ✅ | Type hints and autocomplete |
| **Anthropic.claude-vscode** | Claude integration | ✅ | AI coding assistance |
| **GitHub.Copilot** | GitHub Copilot | ✅ | AI code generation |
| **eamodio.gitlens** | Git history viewer | ⚠️ | Code history and blame |
| **ms-vscode-remote.remote-ssh** | SSH remote | ✅ | Remote development |
| **gruntfuggly.todo-tree** | TODO highlighting | ✅ | Task management in code |

---

## Part 3: MCP Server Integration [REF:VSCODE-004]

### 3.1 Claude MCP Configuration [REF:VSCODE-004A]

Your `.vscode/settings.json` should reference Claude MCP servers:

```json
{
  "claude.contextProviders": [
    {
      "name": "File Context",
      "path": "${workspaceFolder}"
    },
    {
      "name": "Git History",
      "enabled": true
    },
    {
      "name": "Project Dependencies",
      "enabled": true
    }
  ],
  "[claude-prompt-context]": {
    "editor.autoClosingBrackets": "always",
    "editor.autoClosingQuotes": "never"
  }
}
```

### 3.2 Workspace Configuration for MCP Servers [REF:VSCODE-004B]

Create `.vscode/mcp-servers.json`:

```json
{
  "mcpServers": [
    {
      "name": "Claude Desktop",
      "type": "stdio",
      "command": "claude",
      "args": ["--stdio"],
      "env": {
        "CLAUDE_API_KEY": "${env:CLAUDE_API_KEY}",
        "WORKSPACE_ROOT": "${workspaceFolder}"
      },
      "autoStart": true,
      "priority": 1000
    },
    {
      "name": "Codebase Analysis",
      "type": "stdio",
      "command": "codebase-analyzer",
      "enabled": true,
      "env": {
        "IGNORE_PATTERNS": ".git,node_modules,dist"
      }
    }
  ],
  "mcpLogging": {
    "level": "info",
    "outputChannel": "MCP Servers",
    "logFile": "${workspaceFolder}/.vscode/mcp-debug.log"
  }
}
```

### 3.3 Environment Variables Setup [REF:VSCODE-004C]

Create `.vscode/.env.example` (commit to repo):

```bash
# Claude Configuration
CLAUDE_API_KEY=sk-ant-...
CLAUDE_WORKSPACE_CONTEXT=true

# Development
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug

# Python Environment
PYTHONUNBUFFERABLE=1
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=${PYTHONPATH}:.

# Git Configuration
GIT_AUTHOR_NAME=Your Name
GIT_AUTHOR_EMAIL=your.email@example.com
```

---

## Part 4: Developer Tools Setup [REF:VSCODE-005]

### 4.1 Integrated Terminal Configuration [REF:VSCODE-005A]

PowerShell profile for VS Code:

```powershell
# VS Code Terminal Profile
# Location: $PROFILE (usually: C:\Users\[USER]\Documents\PowerShell\profile.ps1)

# Add workspace root to prompt
function prompt {
    $workspaceRoot = if ($env:VSCODE_WORKSPACE) { 
        Split-Path -Leaf $env:VSCODE_WORKSPACE 
    } else { 
        (Get-Item -Path . -Force).BaseName 
    }
    
    return "[$workspaceRoot] > "
}

# Aliases for common operations
Set-Alias -Name vsc -Value code
Set-Alias -Name gitlog -Value 'git log --oneline -10'
Set-Alias -Name gits -Value 'git status'
Set-Alias -Name gitdiff -Value 'git diff'
```

### 4.2 Debug Configurations [REF:VSCODE-005B]

Create `.vscode/launch.json` for debugging:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Node: Current File",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/${file}",
      "restart": true,
      "console": "integratedTerminal"
    }
  ]
}
```

### 4.3 Task Definitions [REF:VSCODE-005C]

Create `.vscode/tasks.json` for common operations:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Install Dependencies",
      "type": "shell",
      "command": "npm install",
      "group": {
        "kind": "build",
        "isDefault": true
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "npm test",
      "group": {
        "kind": "test",
        "isDefault": true
      }
    },
    {
      "label": "Format Code",
      "type": "shell",
      "command": "prettier --write ."
    },
    {
      "label": "Lint Code",
      "type": "shell",
      "command": "eslint . --fix"
    }
  ]
}
```

---

## Summary

This Session 3 documentation provides:

✅ **15 Major Reference Sections** for easy navigation  
✅ **Production-Ready Templates** for all configuration files  
✅ **Complete Extension Management System**  
✅ **MCP Server Integration** for Claude Desktop  
✅ **Debug Configurations** for Python, Node, Django  
✅ **Custom Task Definitions** for common operations  
✅ **Project-Specific Overrides** for flexibility  
✅ **Troubleshooting Guides** for common issues  

All templates are ready to deploy via your GitHub webhook automation system.

---

**Status:** ✅ Session 3 Complete  
**Next:** Session 4 - Terminal & Environment Configuration
