# Example Projects

This document contains runnable example projects to validate the initialization script and project scaffolding.

## Node Example

Location: `examples/node-app`

Commands:

```powershell
cd examples/node-app
npm install
npm start
```

The app is a tiny Express server listening on `3000`.

## Python Example

Location: `examples/python-app`

Commands:

```powershell
cd examples/python-app
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
python app.py
```

The app is a tiny Flask server listening on `5000`.
