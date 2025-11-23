Autonomy Check Script
=====================

This directory contains `autonomy_check.py`, a script used by CI to enforce the
repo's autonomy policy.

Usage examples:

- Dry-run:

  ```bash
  python scripts/autonomy_check.py --dry-run
  ```

- Attempt to merge a PR if allowed by policy (requires `GITHUB_TOKEN`):

  ```bash
  GITHUB_TOKEN=${{ secrets.GITHUB_TOKEN }} python scripts/autonomy_check.py --merge-if-allowed
  ```

For CI, the pipeline installs dependencies and runs the script with the
appropriate `AUTONOMY_LEVEL` value.
