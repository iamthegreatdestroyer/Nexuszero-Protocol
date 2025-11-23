#!/usr/bin/env python3
"""Small autonomy-check script to validate changed files and run tests.

Checks `autonomy.yaml` rules and can optionally merge PRs. Designed to be
used from GitHub Actions or locally.
"""
import argparse
import os
import re
import subprocess
import sys
import json
from typing import List, Dict, Any

try:
    import yaml
except Exception:
    yaml = None

try:
    import requests
except Exception:
    requests = None


def read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_changed_files() -> List[str]:
    """Return list of files changed in this branch vs its base.

    We try a few strategies to stay robust in actions and local runs.
    """
    # Prefer GITHUB_BASE_REF if available
    base_ref = os.environ.get("GITHUB_BASE_REF")
    # head_ref not needed but kept for future use
    if base_ref:
        # fetch base and compare
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", base_ref],
            check=False,
        )
        try:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "FETCH_HEAD...HEAD"],
                encoding="utf-8",
            )
        except subprocess.CalledProcessError:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
                encoding="utf-8",
            )
        files = [p.strip() for p in out.splitlines() if p.strip()]
        if files:
            return files

    # fallback for known PR refs
    ref = os.environ.get("GITHUB_REF") or os.environ.get("BRANCH")
    if ref and ref.startswith("refs/pull/"):
        # refs/pull/123/merge
        try:
            # fetch main and compute diff against it
            _ = re.match(r"refs/pull/(\\d+)/", ref)
            subprocess.check_output(
                ["git", "fetch", "--depth=1", "origin", "main"],
                encoding="utf-8",
            )
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "origin/main...HEAD"],
                encoding="utf-8",
            )
            files = [p.strip() for p in out.splitlines() if p.strip()]
            if files:
                return files
        except Exception:
            pass

    # final fallback: diff from origin/main
    try:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", "main"],
            check=False,
        )
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            encoding="utf-8",
        )
    except subprocess.CalledProcessError:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1...HEAD"], encoding="utf-8"
        )
    return [p.strip() for p in out.splitlines() if p.strip()]


def pattern_matches(pattern: str, path: str) -> bool:
    from fnmatch import fnmatch

    # handle pattern that ends with '/**' meaning prefix match
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return (
            path == prefix
            or path.startswith(prefix + os.sep)
            or path.startswith(prefix + "/")
        )
    if pattern == "**/*" or pattern == "**":
        return True
    return fnmatch(path, pattern)


def allowed_paths_for_level(config: Dict[str, Any], level: int) -> List[str]:
    patterns = []
    allowed = config.get("autonomy", {}).get("allowed_files_by_level", {})
    for lvl_str, pats in allowed.items():
        try:
            lvl = int(lvl_str)
        except ValueError:
            continue
        if lvl <= level and isinstance(pats, list):
            patterns.extend(pats)
    # unique
    return list(dict.fromkeys(patterns))


def check_files_allowed(
    files: List[str], allowed_patterns: List[str]
) -> List[str]:
    disallowed = []
    for f in files:
        ok = False
        for p in allowed_patterns:
            if pattern_matches(p, f):
                ok = True
                break
        if not ok:
            disallowed.append(f)
    return disallowed


def run_tests(commands: List[str]) -> bool:
    for cmd in commands:
        args = cmd if isinstance(cmd, list) else cmd.split()
        print(f"Running: {cmd}")
        rc = subprocess.run(args).returncode
        if rc != 0:
            print(f"Test command failed: {cmd} -> rc={rc}")
            return False
    return True


def get_pr_number() -> int:
    # Use GITHUB_REF like refs/pull/<num>/merge
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/pull/"):
        m = re.match(r"refs/pull/(\d+)/", ref)
        if m:
            return int(m.group(1))
    # try event payload
    gh_event_path = os.environ.get("GITHUB_EVENT_PATH")
    if gh_event_path and os.path.exists(gh_event_path):
        with open(gh_event_path, "r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
                pr = payload.get("pull_request", {})
                if pr and pr.get("number"):
                    return int(pr.get("number"))
            except Exception:
                pass
    # else try environment variable
    pr = os.environ.get("PR_NUMBER")
    if pr and pr.isdigit():
        return int(pr)
    return 0


def gh_automerge(owner: str, repo: str, pr_number: int, token: str) -> bool:
    if requests is None:
        raise RuntimeError(
            "requests not installed; set up the environment or install deps"
        )

    url = (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/pulls/{pr_number}/merge"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"merge_method": "merge"}
    r = requests.put(url, headers=headers, json=payload)
    print("merge response", r.status_code, r.text)
    if r.status_code in (200, 201):
        return True
    return False


def run_check(args):
    config = read_yaml("autonomy.yaml")
    level = int(
        os.environ.get(
            "AUTONOMY_LEVEL",
            config.get("autonomy", {}).get("default_level", 4),
        )
    )
    print(f"Autonomy level: {level}")

    files = get_changed_files()
    print("Changed files:", files)
    allowed_patterns = allowed_paths_for_level(config, level)
    print("Allowed patterns:", allowed_patterns)
    disallowed = check_files_allowed(files, allowed_patterns)
    if disallowed:
        print("Disallowed file changes detected:")
        for p in disallowed:
            print(" -", p)
        return 2

    test_cmds = config.get("autonomy", {}).get("tests_required", [])
    print("Required tests:", test_cmds)
    if test_cmds:
        ok = run_tests(test_cmds)
        if not ok:
            print("Tests failed; aborting autonomy check.")
            return 3

    # At this point, everything looks good.
    print("Autonomy check passed.")

    if args.merge_if_allowed:
        # read config-controlled auto_merge
        if not config.get("autonomy", {}).get("auto_merge", False):
            print("auto_merge disabled in autonomy.yaml; skipping merge.")
            return 0
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            print("GITHUB_TOKEN not set; cannot merge.")
            return 0
        pr_num = get_pr_number()
        if pr_num == 0:
            print("PR number not detected; skipping merge.")
            return 0

        owner_repo = os.environ.get("GITHUB_REPOSITORY")
        if not owner_repo or "/" not in owner_repo:
            print("GITHUB_REPOSITORY not detected; cannot auto-merge.")
            return 0
        owner, repo = owner_repo.split("/", 1)
        print(f"Attempt auto-merge PR #{pr_num} for {owner}/{repo}")
        success = gh_automerge(owner, repo, pr_num, token)
        if success:
            print("PR merged successfully.")
            return 0
        print("Automerge failed.")
        return 4

    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--merge-if-allowed",
        action="store_true",
        help="Attempt to auto-merge if allowed by policy.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No merging; just report status.",
    )
    args = p.parse_args()
    rc = run_check(args)
    if args.dry_run:
        print("Dry-run mode; exit code was", rc)
        # For dry-run, keep exit code 0
        sys.exit(0)
    sys.exit(rc)
