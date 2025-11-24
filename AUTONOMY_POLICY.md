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

This repository is configured to operate by default at **Level 4**.

## Criteria for Automatic Changes

Autonomous changes must:

- Pass all tests and required checks defined in `autonomy.yaml`.
- Follow linting and code style (black/flake8/mypy) whenever applicable.
- Include or update tests to cover functional changes.
- Not leak or persist secrets (explicit check).
- Not reduce coverage below the configured threshold.

## Approvals & Merge Rules

- At Level 4, PRs are eligible for automatic merging when all CI checks pass, no
  policy violations are detected, and no disallowed file changes are included.
- `must_confirm` paths may require explicit owner approval regardless of Level.

## Traceability & Rollback

- Every autonomous change generates a PR with comprehensive description and
  a changelog entry.
- Changes touching sensitive areas may trigger an automatic revert if post-merge
  tests detect regressions.

## How to Modify the Policy

Edit `autonomy.yaml` and create a PR. Autonomous systems should not modify
`autonomy.yaml` without explicit human owner approval unless operating at Level 4
and the change doesn’t touch `must_confirm` patterns.

---

This file is intentionally lightweight and intended for humans; modify `autonomy.yaml`
to adjust enforcement and fine-grained rules.
