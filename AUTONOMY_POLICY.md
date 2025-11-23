# AUTONOMY POLICY

This repository implements an Autonomy Policy to allow the Copilot agent to make
high-confidence, high-value changes autonomously.

This file is a human-readable summary of the machine-readable `autonomy.yaml`.

## Objective

Maximize autonomy while preserving safety, traceability, and quality. Changes
made autonomously should preserve the project's core principles and objectives.

## Autonomy Levels

- Level 0 — No automation: Propose only, no commit or PR
- Level 1 — Conservative: Create branches + PRs, no merge
- Level 2 — Moderate: Auto-merge low-impact changes (docs, tests)
- Level 3 — High: Auto-merge standard features if CI passes
- Level 4 — Full: Auto-merge & merge changes including infra and workflows when safety checks pass

This repository is configured to operate by default at **Level 2**.

## Criteria for Automatic Changes

Autonomous changes must:

- Pass all tests and required checks defined in `autonomy.yaml`.
- Follow linting and code style (black/flake8/mypy) whenever applicable.
- Include or update tests to cover functional changes.
- Not leak or persist secrets (explicit check).
- Not reduce coverage below the configured threshold.

## Approvals & Merge Rules

At Level 2, PRs are eligible for automatic merging for low-impact changes (docs, tests) when CI checks pass and no
policy violations are detected, and no disallowed file changes are included.

- `must_confirm` paths may require explicit owner approval regardless of Level.

## Traceability & Rollback

- Every autonomous change generates a PR with comprehensive description and
  a changelog entry.
- Changes touching sensitive areas may trigger an automatic revert if post-merge
  tests detect regressions.

## How to Modify the Policy

Edit `autonomy.yaml` and create a PR. Autonomous systems should not modify
`autonomy.yaml` without explicit human owner approval unless operating at Level 2
and the change doesn’t touch `must_confirm` patterns.

---

This file is intentionally lightweight and intended for humans; modify `autonomy.yaml`
to adjust enforcement and fine-grained rules.

## Implementation details (autonomy_check)

- `scripts/autonomy_check.py` implements the enforcement logic and several
  reliability improvements:
  - Idempotent CheckRun posting (updates existing checks instead of creating
    duplicates) with retry/backoff on transient errors.
  - Idempotent PR comment behavior (updates owned bot comments with a marker).
  - Merge rule evaluation: polls for required reviews and required checks
    (supports both modern `check-runs` and legacy commit `statuses`).
  - Writes `autonomy-summary-<PR|SHA>.json` artifact for CI traceability.
  - `--skip-heavy-tests` fast path for local iteration and `--merge-if-allowed`
    to perform auto-merge under the configured rules.

These behaviors are designed to keep automation robust and traceable while
minimizing the need for human approvals for safe, policy-compliant changes.

### Local testing & validation

To test the autonomy check locally (fast path):

```pwsh
powershell -ExecutionPolicy Bypass -File .\scripts\setup-python-env.ps1
python scripts/autonomy_check.py --dry-run --skip-heavy-tests --post-check --post-comment
```

For a full check (privileged), run the same command inside a `pull_request_target`
context (for CI) or a safe environment with `GITHUB_TOKEN` set, then run:

```pwsh
# Example in CI (privileged):
python scripts/autonomy_check.py --merge-if-allowed --post-check --post-comment
```
