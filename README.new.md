# Nexuszero Protocol

[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)]() [![Sessions](https://img.shields.io/badge/Sessions%20Complete-4%2F6-blue)]() [![Documentation](https://img.shields.io/badge/Documentation-Complete-success)]() [![License](https://img.shields.io/badge/License-MIT-informational)]()

> A comprehensive VS Code workspace configuration system with automated terminal environment setup, integrated with Claude AI and GitHub automation for zero-friction developer onboarding.

---

## 📖 Overview

Nexuszero Protocol is a production-ready developer environment automation system designed to eliminate manual setup friction. It provides:

- **🎯 Zero-Friction Onboarding** — New developers can clone and run a single command to get started
- **🔧 Unified Configuration** — Consistent settings across projects and team members
- **🤖 AI Integration** — Claide/Claude, Copilot and automation hooks for developer assistance
- **📦 Environment Management** — `.env` patterns with local overrides
- **⚡ PowerShell Automation** — Productivity aliases and helper functions
- **🔄 Auto-Initialization** — Automated detection and setup for Node.js and Python
- **🚀 GitHub Integration** — Actions and webhook automation ready to go

---

## 🚀 Quick Start

### For New Developers

```powershell
# 1. Clone the repo
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# 2. Run the initialization script
.\scripts\init-project.ps1

# 3. Verify
project-info  # View project details
```

### Install the PowerShell profile (Windows)

```powershell
# 1. Find your profile
$PROFILE

# 2. Create the profile directory if needed
$profileDir = Split-Path $PROFILE
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force
}

# 3. Edit the profile and paste the contents from docs/SESSION_4_TERMINAL_ENVIRONMENT.md
notepad $PROFILE

# 4. Reload the profile
& $PROFILE

# 5. Test it
help-vscode
```

---

## 📁 Project Structure

```text
Nexuszero-Protocol/
├── docs/
├── scripts/
│   ├── init-project.ps1
│   ├── setup-python-env.ps1
│   └── setup-node-env.ps1
├── .vscode/
├── .env.example
├── requirements.txt
├── package.json
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🎯 Key Features

- Production ready VS Code workspace settings and recommended extensions
- Terminal automation (PowerShell) and aliases
- Cross-language initialization (Node + Python)
- Secure dev `.env` pattern and local overrides
- Example apps for quick smoke testing

### .vscode configuration snapshot

```text
.vscode/
├── settings.json
├── extensions.json
├── launch.json
└── tasks.json
```

---

## 🔧 Installation & Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol
```

### Step 2 — Install PowerShell profile (Windows)

```powershell
# Replace the next command with the content from docs/SESSION_4_TERMINAL_ENVIRONMENT.md
notepad $PROFILE
```

### Step 3 — Initialize the project

```powershell
# Auto-detect and initialize the project environment
.\scripts\init-project.ps1

# Or specify type
.\scripts\init-project.ps1 -ProjectType "node"
.\scripts\init-project.ps1 -ProjectType "python"
```

### Step 4 — Verify Setup

```powershell
# Helpful commands
help-vscode
project-info
venv
ni  # npm install (if Node project)
```

---

## 🚀 Example Apps

### Node.js example

```powershell
cd examples\node-app
npm install
npm start
```

Visit http://localhost:3000 to validate the Node example.

### Python example

```powershell
cd examples\python-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Visit http://localhost:5000 to validate the Flask example.

---

## 🛠️ Troubleshooting

If your PowerShell profile didn't load or `.env` values didn't apply:

```powershell
# Check Execution Policy (Windows)
Get-ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Check the profile
Test-Path $PROFILE
notepad $PROFILE

# Copy sample .env
copy .env.example .env.local
```

---

## 📚 Documentation

All documentation is in the `/docs/` folder; example sessions include guides for VS Code and Terminal setup.

---

## 🤝 Contributing

Contributions are welcome — please open a PR describing your enhancement and add tests if applicable.

---

## 📄 License

MIT License — see the LICENSE file for details.

---

## ✅ Try the Example Apps

1. Initialize both example environments:

```powershell
.\scripts\init-project.ps1 -ProjectType both
```

2. Run Node example in one terminal:

```powershell
cd examples\node-app
npm install
npm start
```

3. Run Python example in a second terminal:

```powershell
cd examples\python-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If everything works, you should be able to access Node on port 3000 and Flask on port 5000.
