# Nexuszero Protocol

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Sessions](https://img.shields.io/badge/Sessions%20Complete-4%2F6-blue)
![Documentation](https://img.shields.io/badge/Documentation-Complete-success)
![Coverage](https://img.shields.io/badge/Coverage-90.48%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-informational)

> A comprehensive VS Code workspace configuration system with automated terminal environment setup, integrated with Claude AI and GitHub automation for zero-friction developer onboarding.

---

## 📖 Overview

Nexuszero Protocol is a production-ready developer environment automation system designed to eliminate manual setup friction. It provides:

- **🎯 Zero-Friction Onboarding** - New developers clone, run one command, everything works
- **🔧 Unified Configuration** - Consistent settings across all projects and team members
- **🤖 AI Integration** - Deep Claude Desktop and GitHub Copilot integration
- **📦 Environment Management** - Secure credential handling with `.env` patterns
- **⚡ PowerShell Automation** - 35+ aliases and 10+ utility functions for productivity
- **🔄 Auto-Initialization** - Automatic detection and setup for Node.js and Python projects
- **🚀 GitHub Integration** - Webhook automation for new repository configuration

---

## 🚀 Quick Start

### For New Developers

```powershell
# 1. Clone the repository
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# 2. Run the initialization script
.\\scripts\\init-project.ps1

# 3. Everything is set up! Start coding
project-info  # View project details
```

### For PowerShell Profile Installation

```powershell
# 1. Find your profile location
$PROFILE

# 2. Create PowerShell directory if needed
$profileDir = Split-Path $PROFILE
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force
}

# 3. Copy the profile content from docs/SESSION_4_TERMINAL_ENVIRONMENT.md
# Paste into your $PROFILE file

# 4. Reload the profile
& $PROFILE

# 5. Test it
help-vscode  # View all available commands
```

---

## 📁 Project Structure

```
Nexuszero-Protocol/
├── docs/                                    # Complete documentation
│   ├── SESSION_3_WORKSPACE_CONFIGURATION.md # VS Code workspace setup
│   ├── SESSION_4_TERMINAL_ENVIRONMENT.md    # Terminal & environment config
│   ├── SESSIONS_3_4_REFERENCE_GUIDE.md      # Quick reference
│   └── CHECKPOINT_SESSION_4.md              # Latest checkpoint
│
├── scripts/                                 # Setup and automation scripts
│   ├── init-project.ps1                     # Universal project initialization
│   ├── setup-python-env.ps1                 # Python environment setup
│   └── setup-node-env.ps1                   # Node.js environment setup
│
├── .vscode/                                 # VS Code workspace configuration
│   ├── settings.json                        # Workspace settings
│   ├── extensions.json                      # Recommended extensions
│   ├── launch.json                          # Debug configurations
│   └── tasks.json                           # Custom task definitions
│
├── .env.example                             # Environment variables template
├── .npmrc                                   # npm configuration
├── requirements.txt                         # Python dependencies
├── package.json                             # Node.js project configuration
├── .gitignore                               # Git exclusion rules
├── README.md                                # This file
└── LICENSE                                  # MIT License
```

---

## 🎯 Key Features

### 1. VS Code Workspace Configuration [REF:VSCODE-001]

**Automatic Setup:**

- ✅ Production-ready settings across all projects
- ✅ 20+ recommended extensions with auto-install
- ✅ Debug configurations for Python, Node.js, Django
- ✅ 13+ custom task definitions
- ✅ Language-specific overrides

**Configuration Files:**

```
.vscode/
├── settings.json       # Editor, formatter, linter, terminal config
├── extensions.json     # Auto-install extensions
├── launch.json         # Debug configurations
├── tasks.json          # Custom tasks
└── keybindings.json    # Keyboard shortcuts
```

**Features:**

- Consistent code formatting (Prettier + Black)
- Automatic ESLint and Pylint integration
- MCP server configuration for Claude Desktop
- Git integration (GitLens)
- Python and JavaScript/TypeScript support

### 2. Terminal & Environment Setup [REF:TERM-001]

**PowerShell Profile (350 lines):**

- ✅ 35+ custom aliases for git, npm, Python, Docker
- ✅ 10+ utility functions
- ✅ Enhanced prompt showing project context
- ✅ Auto-loading environment variables
- ✅ Virtual environment management
- ✅ Git integration and helpers

**Quick Alias Reference:**

```powershell
# Git operations
gs = git status
ga = git add
gc = git commit
gp = git push
gl = git log --oneline -10
gb = git branch
gco = git checkout

# Project management
project-info        # Show project details
init-project       # Initialize project environment
venv              # Activate Python virtual environment
make-venv         # Create new virtual environment

# VS Code
vsc = code
vsca = code .      # Open current directory
vsced = code $PROFILE  # Edit PowerShell profile

# Node.js
ni = npm install
nr = npm run
nrd = npm run dev
nrt = npm run test
npm-scripts       # List available npm scripts

# Navigation
ll = List files (with hidden)
home = Go to home directory
dev = Go to ~/Development
```

### 3. Environment Variable Management [REF:TERM-003]

**Secure Pattern:**

- `.env.example` - Committed to repository (template)
- `.env.local` - Git-ignored (local configuration)
- Auto-loaded by PowerShell profile

**Categories (40+ variables):**

```bash
# Claude Configuration
CLAUDE_API_KEY=sk-ant-...
CLAUDE_WORKSPACE_CONTEXT=true
CLAUDE_MODEL=claude-opus-4-1

# GitHub Integration
GITHUB_TOKEN=ghp_...
GITHUB_USERNAME=your-username

# Development
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug

# Python
PYTHONUNBUFFERABLE=1
PYTHONDONTWRITEBYTECODE=1

# Database
DATABASE_URL=postgres://...
REDIS_URL=redis://...

# Server
PORT=3000
HOST=0.0.0.0
API_URL=http://localhost:3000

# AWS (Optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Feature Flags
ENABLE_TESTING=true
ENABLE_LOGGING=true
ENABLE_METRICS=false
```

### 4. Automated Initialization [REF:TERM-006]

**Universal Init Script:**

- Auto-detects project type (Node.js, Python, hybrid)
- Installs dependencies automatically
- Creates Python virtual environment if needed
- Initializes environment variables
- Sets up development environment

**Usage:**

```powershell
# Auto-detect and initialize
.\\scripts\\init-project.ps1

# Specify project type
.\\scripts\\init-project.ps1 -ProjectType \"node\"
.\\scripts\\init-project.ps1 -ProjectType \"python\"
.\\scripts\\init-project.ps1 -ProjectType \"both\"

# Test without making changes
.\\scripts\\init-project.ps1 -DryRun
```

### 5. GitHub Webhook Integration

**Automatic Repository Configuration:**

- New repositories automatically receive:
- VS Code workspace settings
- PowerShell profile guidance
- Setup scripts
- Environment variable templates
- GitHub Actions workflows (Session 5)
- MCP configuration (Session 5)

**Result:**

- New developers clone → Run `init-project` → Everything works
- Zero manual setup required
- Consistent environment across team

---

## 📚 Documentation

All documentation is in the `/docs/` directory:

### [SESSION_3_WORKSPACE_CONFIGURATION.md](docs/SESSION_3_WORKSPACE_CONFIGURATION.md)

Complete guide to VS Code workspace setup with 15 major reference sections:

- Core workspace settings [REF:VSCODE-002]
- Extension management [REF:VSCODE-003]
- MCP server integration [REF:VSCODE-004]
- Developer tools [REF:VSCODE-005]
- Project overrides [REF:VSCODE-006]
- And 10 more sections with full details

### [SESSION_4_TERMINAL_ENVIRONMENT.md](docs/SESSION_4_TERMINAL_ENVIRONMENT.md)

Complete guide to terminal and environment setup with 11 major reference sections:

- PowerShell profile setup [REF:TERM-002]
- Environment variables [REF:TERM-003]
- Python environment [REF:TERM-004]
- Node.js setup [REF:TERM-005]
- Auto-initialization [REF:TERM-006]
- And 6 more sections with full details

### [SESSIONS_3_4_REFERENCE_GUIDE.md](docs/SESSIONS_3_4_REFERENCE_GUIDE.md)

Quick reference guide with:

- Complete reference code map (26 sections)
- Quick command reference (35+ aliases)
- Implementation checklist
- File structure guide
- Environment variables quick ref

### [CHECKPOINT_SESSION_4.md](docs/CHECKPOINT_SESSION_4.md)

Session 4 checkpoint with:

- What's been completed
- How to continue
- Implementation status
- Next steps for Session 5

---

## 🧪 Testing & Quality

**Current Coverage:** 90.48% (Tarpaulin, +1.03% from 89.45%) – target ≥90% achieved.

**Test Assets:** Structured JSON test vectors (LWE, Ring-LWE, Proof) using unified schema documented in `docs/TEST_VECTORS_SCHEMA.md`.

**Implemented Suites:**

- Unit tests (encryption, polynomial ops, NTT, proofs)
- Property-based tests (LWE, proofs)
- Negative tests (validation, tampering, mismatches) – **60 tests passing**
- Benchmarks (LWE encrypt, proof gen/verify)
- Test vector ingestion + execution (`tests/test_vector_runner.rs`)

**Recent Improvements:**

- Statement module: 48.1% → 90.7% (+42.6%) via validation tests
- Witness module: 63.8% → 81.0% (+17.2%) via boundary & mismatch tests
- Proof module: 75.1% → 83.6% (+8.5%) via tampering & edge-case verification tests

**Next Target:** Strengthen statistical distribution tests, advanced range proof implementation, and performance regression tracking.

---

## 🔧 Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol
```

### Step 2: Install PowerShell Profile

**Windows:**

```powershell
# Find profile location
$PROFILE  # Usually: C:\\Users\\[USER]\\Documents\\PowerShell\\profile.ps1

# Create directory if needed
$profileDir = Split-Path $PROFILE
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force
}

# Copy content from SESSION_4_TERMINAL_ENVIRONMENT.md [REF:TERM-002B]
# Open in editor and paste
notepad $PROFILE

# Set execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Reload profile
& $PROFILE
```

### Step 3: Initialize Project

```powershell
# Auto-detect and setup
.\\scripts\\init-project.ps1

# Or specify type
.\\scripts\\init-project.ps1 -ProjectType \"node\"
.\\scripts\\init-project.ps1 -ProjectType \"python\"
```

### Step 4: Verify Setup

```powershell
# Test PowerShell commands
help-vscode      # View all commands
project-info     # Show project details
gs              # git status
gs              # npm scripts (if Node project)
venv            # Python virtual environment (if Python project)
```

---

## 🚀 Usage Examples

### Example 1: Node.js Project

```powershell
# Clone and initialize
git clone <your-node-project>
cd your-node-project
.\\scripts\\init-project.ps1

# Now available:
ni              # Install dependencies
nr dev          # Run dev server
nrt             # Run tests
npm-scripts     # List scripts
gs              # git status
```

### Example 2: Python Project

```powershell
# Clone and initialize
git clone <your-python-project>
cd your-python-project
.\\scripts\\init-project.ps1

# Now available:
venv            # Activate virtual environment
project-info    # View project details
gs              # git status

# Run Python
python script.py
```

### Example 3: Hybrid Project

```powershell
# Clone and initialize
git clone <hybrid-project>
cd hybrid-project
.\\scripts\\init-project.ps1

# Both Node and Python environments ready
ni              # npm install
venv            # Python venv activated
nr dev          # Run Node dev server
```

---

## 🔄 Workflow Integration

### VS Code Integration

**Automatic Benefits:**

- Extensions auto-install when you open the project
- Settings apply instantly
- Debug configurations ready to use
- Terminal opens with PowerShell profile loaded
- Tasks available via `Ctrl+Shift+P` → Tasks

### Git Workflow

**PowerShell Aliases Make Git Fast:**

```powershell
gs              # Check status
ga .            # Add changes
gc \"message\"    # Commit
gp              # Push
gl              # View recent commits
git-log-graph   # Visual history
git-cleanup     # Delete merged branches
```

### Development Cycle

```powershell
# 1. Start day
vscode          # Open VS Code
project-info    # Check project status

# 2. Make changes
venv            # Activate if Python
nr dev          # Run dev server

# 3. Test changes
nrt             # Run tests

# 4. Commit
gs              # Check status
ga .            # Stage changes
gc \"Fix: describe change\"  # Commit
gp              # Push
```

---

## 📊 Project Statistics

| Metric                  | Count  | Status        |
| ----------------------- | ------ | ------------- |
| Documentation Lines     | 6,300+ | ✅ Complete   |
| Reference Codes         | 26     | ✅ Complete   |
| Configuration Templates | 12+    | ✅ Ready      |
| Setup Scripts           | 3      | ✅ Production |
| PowerShell Aliases      | 35+    | ✅ Active     |
| Utility Functions       | 10+    | ✅ Active     |
| Environment Variables   | 40+    | ✅ Documented |
| Sessions Complete       | 4/6    | ✅ On Track   |

---

## 🗺️ Project Roadmap

### ✅ Completed (Sessions 1-4)

- Session 1: Memory system & context
- Session 2: GitHub webhook automation review
- Session 3: VS Code workspace configuration (15 sections)
- Session 4: Terminal & environment setup (11 sections)

### 🚀 In Progress (Session 5)

- MCP server deep configuration
- GitHub Actions CI/CD workflows
- Testing automation integration
- Deployment validation

### 📋 Planned (Session 6)

- Webhook integration finalization
- New repository setup verification
- Monitoring & logging system
- Complete troubleshooting guide

---

## 🤝 Contributing

Contributions are welcome! This project is actively being developed.

**How to contribute:**

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request with description

**Areas for contribution:**

- Additional language support (Go, Rust, Java)
- More VS Code extensions recommendations
- Enhanced error handling
- Documentation improvements
- Additional automation scripts

---

## 🐛 Troubleshooting

### PowerShell Profile Not Loading

```powershell
# Check execution policy
Get-ExecutionPolicy

# Set if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify profile exists
Test-Path $PROFILE

# Reload
& $PROFILE
```

### Environment Variables Not Loading

```powershell
# Check if .env.local exists
Test-Path \".env.local\"

# Create from example if needed
copy .env.example .env.local

# Edit with your values
notepad .env.local

# Reload terminal
& $PROFILE
```

### Virtual Environment Issues

```powershell
# Check if .venv exists
Test-Path \".venv\"

# Create new virtual environment
make-venv

# Activate
venv

# Install requirements
pip install -r requirements.txt
```

### VS Code Extensions Not Installing

1. Close VS Code completely
2. Delete `.vscode/` if you want fresh install
3. Clone again or pull latest
4. Open VS Code
5. Extensions will auto-install

For detailed troubleshooting, see:

- `docs/SESSION_3_WORKSPACE_CONFIGURATION.md` [REF:VSCODE-011]
- `docs/SESSION_4_TERMINAL_ENVIRONMENT.md` [REF:TERM-009]

---

## 📞 Support & Documentation

**Full Documentation:**

- [VS Code Workspace Configuration](docs/SESSION_3_WORKSPACE_CONFIGURATION.md)
- [Terminal & Environment Setup](docs/SESSION_4_TERMINAL_ENVIRONMENT.md)
- [Reference Guide](docs/SESSIONS_3_4_REFERENCE_GUIDE.md)
- [Session 4 Checkpoint](docs/CHECKPOINT_SESSION_4.md)

**Quick Reference:**

- Type `help-vscode` in PowerShell for command list
- Type `project-info` to see your project details
- See docs for detailed setup instructions

**Issues:**
If you encounter issues:

1. Check troubleshooting section above
2. Review relevant documentation
3. Check if .env.local is properly configured
4. Verify PowerShell profile is loaded

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Steve (iamthegreatdestroyer)**

- GitHub: [@iamthegreatdestroyer](https://github.com/iamthegreatdestroyer)
- Project: [Nexuszero Protocol](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git)

---

## 🙏 Acknowledgments

Built with:

- **Claude AI** (Anthropic) - AI-powered development assistance
- **GitHub Copilot** - Code completion
- **VS Code** - Development environment
- **PowerShell** - Automation and scripting

---

## 📈 Project Status

```
\u2705 Documentation: COMPLETE (6,300+ lines)
\u2705 Core Features: COMPLETE (VS Code + Terminal + Environment)
\u2705 GitHub Integration: READY (Scripts prepared)
\ud83d\udea7 Session 5: IN PROGRESS (MCP & GitHub Actions)
\ud83d\udccb Session 6: PLANNED (Final deployment & monitoring)
```

**Last Updated:** November 21, 2025
**Current Session:** 4/6
**Status:** 🟢 Active Development

---

## 🎯 Get Started Now

```bash
# 1. Clone
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# 2. Initialize
.\\scripts\\init-project.ps1

# 3. Verify
project-info

# 4. Start coding!
```

**Questions?** Check the documentation in `/docs/` or use `help-vscode` after profile installation.

---

**Made with ❤️ for developers who value productivity and automation**
"
