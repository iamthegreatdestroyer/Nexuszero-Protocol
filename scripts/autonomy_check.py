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
import time
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
    # try a few times for transient errors
    for attempt in range(3):
        try:
            r = requests.put(url, headers=headers, json=payload)
        except Exception as e:
            print("merge request exception", e)
            r = None
        if r is not None and r.status_code in (200, 201):
            print("merge response", r.status_code, r.text)
            return True
        if r is not None:
            print("merge attempt failed", attempt, r.status_code, r.text)
        time.sleep(1 + attempt * 2)
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

    # Support a light fast test set if requested (--skip-heavy-tests)
    fast_cmds = config.get("autonomy", {}).get("fast_tests_required")
    test_cmds = (
        fast_cmds if args.skip_heavy_tests and fast_cmds is not None
        else config.get("autonomy", {}).get("tests_required", [])
    )
    print("Required tests:", test_cmds)
    if test_cmds:
        ok = run_tests(test_cmds)
        if not ok:
            print("Tests failed; aborting autonomy check.")
            return 3

    # At this point, everything looks good.
    print("Autonomy check passed.")

    # Optionally produce a JSON summary used by CI for artifact uploads
    def write_summary_file(summary_obj: Dict[str, Any]):
        fname = "autonomy-summary-{}.json".format(
            os.environ.get("PR_NUMBER")
            or os.environ.get("GITHUB_SHA")
            or "summary"
        )
        try:
            with open(fname, "w", encoding="utf-8") as fh:
                json.dump(summary_obj, fh, indent=2)
            print("Wrote autonomy summary to", fname)
        except Exception as e:
            print("Failed to write summary file", e)

    # Optionally post a check-run and/or a PR comment for UI visibility
    if getattr(args, "post_check", False):
        owner_repo = os.environ.get("GITHUB_REPOSITORY")
        token = os.environ.get("GITHUB_TOKEN")
        sha = os.environ.get("GITHUB_SHA") or ""
        if owner_repo and token and sha:
            owner, repo = owner_repo.split("/", 1)
            title = f"Autonomy check (level {level})"
            summary = f"Autonomy check passed for files: {files}"
            ok = post_check_run(
                owner, repo, sha, token, title, summary, "success"
            )
            print("Posted check-run?", ok)
        else:
            print(
                "Skipping check-run post, missing GITHUB_TOKEN, "
                "GITHUB_REPOSITORY or GITHUB_SHA"
            )

    if getattr(args, "post_comment", False):
        owner_repo = os.environ.get("GITHUB_REPOSITORY")
        token = os.environ.get("GITHUB_TOKEN")
        pr_num = get_pr_number()
        if owner_repo and token and pr_num:
            owner, repo = owner_repo.split("/", 1)
            body = (
                f"AUTONOMY SUMMARY\n\nFiles: {files}\n\nLevel: {level}\n"
            )
            ok = post_or_update_pr_comment(
                owner, repo, pr_num, token, body, marker="AUTONOMY SUMMARY"
            )
            print("Posted PR comment?", ok)
        else:
            print(
                "Skipping PR comment, missing GITHUB_TOKEN, "
                "GITHUB_REPOSITORY or PR_NUMBER"
            )

    # Write summary artifact for CI
    try:
        summary_obj = {"files": files, "level": level}
        write_summary_file(summary_obj)
    except Exception:
        pass

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
        merge_rules = config.get("autonomy", {}).get("merge_rules", {})
        if not pr_meets_merge_rules(owner, repo, pr_num, token, merge_rules):
            print("PR does not meet merge rules; aborting auto-merge.")
            return 0
        success = gh_automerge(owner, repo, pr_num, token)
        if success:
            print("PR merged successfully.")
            return 0
        print("Automerge failed.")
        return 4

    return 0


def post_check_run(
    owner: str,
    repo: str,
    sha: str,
    token: str,
    title: str,
    summary: str,
    conclusion: str = "success",
) -> bool:
    """Create a GitHub check run to surface results in PR UI.
    Minimal implementation; in CI it requires checks: write permissions.
    """
    if requests is None:
        raise RuntimeError(
            "requests not installed; install `requests` in the environment"
        )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # If an existing check run with the same name exists for this SHA,
    # update it to avoid duplicates in the PR UI.
    # This avoids spamming the PR UI with repeated checks.
    list_url = (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/commits/{sha}/check-runs"
    )
    r = requests.get(list_url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        for cr in data.get("check_runs", []):
            if cr.get("name") == title:
                # update
                update_url = (
                    "https://api.github.com/repos/"
                    f"{owner}/{repo}/check-runs/{cr.get('id')}"
                )
                payload = {
                    "name": title,
                    "head_sha": sha,
                    "status": "completed",
                    "conclusion": conclusion,
                    "output": {"title": title, "summary": summary},
                }
                # PATCH with retry
                for attempt in range(3):
                    try:
                        r2 = requests.patch(
                            update_url,
                            headers=headers,
                            json=payload,
                        )
                    except Exception as e:
                        print("check-run update exception", e)
                        r2 = None
                    if r2 is not None and r2.status_code in (200, 201):
                        print("check-run update response", r2.status_code)
                        return True
                    if r2 is not None:
                        print(
                            "check-run update attempt failed",
                            attempt,
                            r2.status_code,
                        )
                    time.sleep(1 + attempt * 2)

    # otherwise create a new check-run
    create_url = (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/check-runs"
    )
    payload = {
        "name": title,
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": title, "summary": summary},
    }
    # POST with retry
    for attempt in range(3):
        try:
            r3 = requests.post(create_url, headers=headers, json=payload)
        except Exception as e:
            print("check-run create exception", e)
            r3 = None
        if r3 is not None and r3.status_code in (200, 201):
            print("check-run create response", r3.status_code)
            return True
        if r3 is not None:
            print(
                "check-run create attempt failed",
                attempt,
                getattr(r3, "status_code", None),
            )
        time.sleep(1 + attempt * 2)
    return False


def post_pr_comment(
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    body: str,
) -> bool:
    if requests is None:
        raise RuntimeError(
            "requests not installed; install `requests` in the environment"
        )
    url = (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/issues/{pr_number}/comments"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"body": body}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload)
        except Exception as e:
            print("post-comment exception", e)
            r = None
        if r is not None and r.status_code in (200, 201):
            print("post-comment", r.status_code)
            return True
        if r is not None:
            print("post-comment attempt failed", attempt, r.status_code)
        time.sleep(1 + attempt * 2)
    return False


def post_or_update_pr_comment(
    owner: str, repo: str, pr_number: int, token: str, body: str, marker: str
) -> bool:
    """Post a new comment or update an existing one that contains 'marker'."""
    if requests is None:
        raise RuntimeError(
            "requests not installed; install `requests` in the environment"
        )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    # Get existing comments
    list_url = (
        "https://api.github.com/repos/"
        f"{owner}/{repo}/issues/{pr_number}/comments"
    )
    r = requests.get(list_url, headers=headers)
    if r.status_code == 200:
        for com in r.json():
            if marker in com.get("body", ""):
                # We only update bot comments or comments that match the
                # marker to avoid editing a user's content
                user_type = com.get("user", {}).get("type")
                if (
                    user_type == "Bot"
                    or user_type == "App"
                    or user_type == "Organization"
                ):
                    cid = com.get("id")
                    patch_url = (
                        "https://api.github.com/repos/"
                        f"{owner}/{repo}/issues/comments/{cid}"
                    )
                    for attempt in range(3):
                        try:
                            r2 = requests.patch(
                                patch_url, headers=headers, json={"body": body}
                            )
                        except Exception as e:
                            print("update-comment exception", e)
                            r2 = None
                        if r2 is not None and r2.status_code in (200, 201):
                            print("update-comment", r2.status_code)
                            return True
                        if r2 is not None:
                            print(
                                "update-comment attempt failed",
                                attempt,
                                getattr(r2, "status_code", None),
                            )
                        time.sleep(1 + attempt * 2)
    # No existing comment matched - create one
    return post_pr_comment(owner, repo, pr_number, token, body)


def pr_meets_merge_rules(
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    rules: Dict[str, Any],
    timeout: int = 300,
) -> bool:
    """Return True if PR meets the merge rules specified in `rules`.

    Rules may include `required_reviewers`, `required_checks`.
    This function will poll check runs up to `timeout` seconds
    for checks to pass.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    # fetch PR to get head SHA
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    r = requests.get(pr_url, headers=headers)
    if r.status_code != 200:
        print("Failed to fetch PR info", r.status_code)
        return False
    pr = r.json()
    head_sha = pr.get("head", {}).get("sha")

    # check reviews
    required_reviews = int(rules.get("required_reviewers", 0))
    if required_reviews > 0:
        reviews_url = (
            "https://api.github.com/repos/"
            f"{owner}/{repo}/pulls/{pr_number}/reviews"
        )
        r2 = requests.get(reviews_url, headers=headers)
        if r2.status_code != 200:
            print("Failed to fetch PR reviews", r2.status_code)
            return False
        approvals = 0
        for rev in r2.json():
            if rev.get("state") == "APPROVED":
                approvals += 1
        if approvals < required_reviews:
            print("Not enough approvals", approvals, required_reviews)
            return False

    # check runs polling
    required_checks = rules.get("required_checks", []) or []
    if required_checks:
        checks_url = (
            "https://api.github.com/repos/"
            f"{owner}/{repo}/commits/{head_sha}/check-runs"
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            r3 = requests.get(checks_url, headers=headers)
            if r3.status_code == 200:
                runs = {
                    cr.get("name"): cr.get("conclusion")
                    for cr in r3.json().get("check_runs", [])
                }
                # also pull commit statuses (contexts) - some CI posts statuses
                status_url = (
                    "https://api.github.com/repos/"
                    f"{owner}/{repo}/commits/{head_sha}/status"
                )
                statuses = {}
                rs = requests.get(status_url, headers=headers)
                if rs.status_code == 200:
                    for s in rs.json().get("statuses", []) or []:
                        statuses[s.get("context")] = s.get("state")
                missing = [
                    c
                    for c in required_checks
                    if (
                        runs.get(c) != "success"
                        and statuses.get(c) != "success"
                    )
                ]
                if not missing:
                    return True
                print("Check run conclusions:", runs)
                print("Commit statuses:", statuses)
                print("Waiting for checks to pass; still missing", missing)
            else:
                print("Failed to fetch check runs", r3.status_code)
            time.sleep(10)
        print("Timeout waiting for required checks to pass")
        return False

    # if no required checks and no reviewer requirements, we meet the rules
    return True


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
    p.add_argument(
        "--skip-heavy-tests",
        action="store_true",
        help=(
            "Run fast tests (if configured) and skip heavy tests like "
            "Rust/PyTorch ones."
        ),
    )
    p.add_argument(
        "--post-check",
        action="store_true",
        help="Post a CheckRun with the autonomy-check result to the PR.",
    )
    p.add_argument(
        "--post-comment",
        action="store_true",
        help="Post a PR comment summarizing the autonomy-check result.",
    )
    args = p.parse_args()
    rc = run_check(args)
    if args.dry_run:
        print("Dry-run mode; exit code was", rc)
        # For dry-run, keep exit code 0
        sys.exit(0)
    sys.exit(rc)
