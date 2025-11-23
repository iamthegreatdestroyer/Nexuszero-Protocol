import subprocess
from unittest.mock import patch, MagicMock


from scripts.autonomy_check import (
    pattern_matches,
    allowed_paths_for_level,
    check_files_allowed,
    get_pr_number,
    run_tests,
    post_check_run,
    post_pr_comment,
    post_or_update_pr_comment,
    pr_meets_merge_rules,
    gh_automerge,
)


def test_pattern_matches_prefix():
    assert pattern_matches(
        "nexuszero-optimizer/**", "nexuszero-optimizer/foo/bar.py"
    )
    assert pattern_matches("**/*", "anything/here.txt")
    assert pattern_matches("docs/**", "docs/README.md")
    assert pattern_matches("README*.md", "README.new.md")


def test_allowed_paths_for_level(tmp_path):
    cfg = {
        "autonomy": {
            "allowed_files_by_level": {
                "1": ["docs/**"],
                "2": ["nexuszero-optimizer/**"],
            }
        }
    }
    pats = allowed_paths_for_level(cfg, 1)
    assert "docs/**" in pats
    pats2 = allowed_paths_for_level(cfg, 2)
    assert "nexuszero-optimizer/**" in pats2


def test_check_files_allowed():
    patt = ["docs/**", "examples/**"]
    files = ["docs/foo.md", "examples/a.py", "src/main.rs"]
    disallowed = check_files_allowed(files, patt)
    assert "src/main.rs" in disallowed


def test_get_pr_number_from_env(monkeypatch):
    monkeypatch.setenv("PR_NUMBER", "123")
    assert get_pr_number() == 123


def test_run_tests_success(monkeypatch):
    # patch subprocess.run to return rc=0
    mock = MagicMock()
    mock.return_value = subprocess.CompletedProcess(
        args=["echo"], returncode=0
    )
    with patch("subprocess.run", mock):
        assert run_tests(["echo hello"]) is True


def test_post_check_run_and_comment(monkeypatch):
    # Mock requests.post
    class FakeResp:
        def __init__(self, code=201):
            self.status_code = code

    def fake_post(url, headers=None, json=None):
        return FakeResp(201)

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    # Patch both post and get endpoints to simulate existing check-run
    # and comments; little mocks for success codes.
    
    def fake_get(url, headers=None):
        # emulate listing check runs: empty for now

        class Fake:
            status_code = 200

            def json(self):
                return {"check_runs": []}

        return Fake()

    with patch("scripts.autonomy_check.requests.post", fake_post), patch(
        "scripts.autonomy_check.requests.get", fake_get
    ):
        ok = post_check_run("owner", "repo", "deadbeef", "token", "t", "s")
        assert ok
        ok2 = post_pr_comment("owner", "repo", 12, "token", "body")
        assert ok2


def test_comment_update(monkeypatch):
    # Simulate fetching comments containing our marker and ensure update uses
    # PATCH
    def fake_get_comments(url, headers=None):
        class FakeResp:
            status_code = 200

            def json(self):
                return [
                    {
                        "id": 1,
                        "body": "AUTONOMY SUMMARY existing",
                        "user": {"type": "Bot"},
                    }
                ]
        return FakeResp()

    def fake_patch(url, headers=None, json=None):
        class FakeR:
            status_code = 200
        return FakeR()

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    with patch(
        "scripts.autonomy_check.requests.get", fake_get_comments
    ), patch("scripts.autonomy_check.requests.patch", fake_patch):
        ok = post_or_update_pr_comment(
            "owner",
            "repo",
            12,
            "token",
            "body",
            marker="AUTONOMY SUMMARY",
        )
        assert ok


def test_comment_update_user_not_updated(monkeypatch):
    # Ensure user comments are not updated by the script
    def fake_get_comments(url, headers=None):
        class FakeResp:
            status_code = 200

            def json(self):
                return [
                    {
                        "id": 3,
                        "body": "AUTONOMY SUMMARY existing",
                        "user": {"type": "User"},
                    }
                ]
        return FakeResp()

    def fake_post(url, headers=None, json=None):
        class FakeR:
            status_code = 201
        return FakeR()

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    with patch(
        "scripts.autonomy_check.requests.get", fake_get_comments
    ), patch(
        "scripts.autonomy_check.requests.post", fake_post
    ):
        ok = post_or_update_pr_comment(
            "owner",
            "repo",
            12,
            "token",
            "body",
            marker="AUTONOMY SUMMARY",
        )
        # Should result in a new comment posted instead of patch
        assert ok


def test_post_check_run_retries(monkeypatch):
    # Simulate requests.post failing first then succeeding
    state = {"calls": 0}

    class FakeResp:
        def __init__(self, code=500):
            self.status_code = code

    def fake_post(url, headers=None, json=None):
        state["calls"] += 1
        if state["calls"] == 1:
            return FakeResp(500)
        return FakeResp(201)

    def fake_get(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"check_runs": []}
        return Fake()

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    with patch("scripts.autonomy_check.requests.post", fake_post), patch(
        "scripts.autonomy_check.requests.get", fake_get
    ):
        ok = post_check_run("owner", "repo", "deadbeef", "token", "t", "s")
        assert ok


def test_pr_meets_merge_rules(monkeypatch):
    owner, repo, pr = "owner", "repo", 12
    # Mock PR response with head.sha
    
    def fake_get_pr(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"head": {"sha": "deadbeef"}}
        return Fake()

    def fake_get_reviews(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return [{"state": "APPROVED"}]
        return Fake()

    def fake_get_check_runs(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"check_runs": [
                    {"name": "pytest", "conclusion": "success"},
                    {"name": "lint", "conclusion": "success"},
                    {"name": "coverage", "conclusion": "success"},
                ]}
        return Fake()

    def fake_get_status(url, headers=None):
        class Fake:
            status_code = 200
            def json(self):
                return {"statuses": []}
        return Fake()

    rules = {
        "required_reviewers": 1,
        "required_checks": ["pytest", "lint", "coverage"],
    }
    with patch(
        "scripts.autonomy_check.requests.get",
        side_effect=[
            fake_get_pr(None),
            fake_get_reviews(None),
            fake_get_check_runs(None),
            fake_get_status(None),
        ],
    ): 
        ok = pr_meets_merge_rules(owner, repo, pr, "token", rules, timeout=5)
        assert ok is True


def test_pr_meets_merge_rules_with_statuses(monkeypatch):
    owner, repo, pr = "owner", "repo", 14

    def fake_get_pr(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"head": {"sha": "deadbeef2"}}
        return Fake()

    def fake_get_reviews(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return [{"state": "APPROVED"}]
        return Fake()

    def fake_get_checks(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"check_runs": []}
        return Fake()

    def fake_get_status(url, headers=None):
        class Fake:
            status_code = 200

            def json(self):
                return {"statuses": [
                    {"context": "pytest", "state": "success"},
                    {"context": "lint", "state": "success"},
                    {"context": "coverage", "state": "success"},
                ]}
        return Fake()

    rules = {
        "required_reviewers": 1,
        "required_checks": ["pytest", "lint", "coverage"],
    }
    with patch("scripts.autonomy_check.requests.get", side_effect=[
        fake_get_pr(None),
        fake_get_reviews(None),
        fake_get_checks(None),
        fake_get_status(None),
    ]):
        ok = pr_meets_merge_rules(owner, repo, pr, "token", rules, timeout=5)
        assert ok is True


def test_summary_file_written(tmp_path, monkeypatch):
    # Create a minimal args object. Ensure no network calls by keeping
    # post-check/comment off and skip heavy tests.
    from types import SimpleNamespace

    args = SimpleNamespace(
        merge_if_allowed=False,
        dry_run=False,
        skip_heavy_tests=True,
        post_check=False,
        post_comment=False,
    )
    # monkeypatch run_tests to avoid executing subprocess
    with patch(
        "scripts.autonomy_check.run_tests", lambda x: True
    ), patch(
        "scripts.autonomy_check.read_yaml", lambda path: {
            "autonomy": {"allowed_files_by_level": {"2": ["**/*"]}}
        },
    ), patch(
        "scripts.autonomy_check.get_changed_files",
        lambda: ["README.md"],
    ):
        monkeypatch.setenv("PR_NUMBER", "999")
        # set working dir to tmp dir
        cwd = tmp_path
        monkeypatch.chdir(str(cwd))
        # call run_check which should write the summary file
        from scripts.autonomy_check import run_check

        run_check(args)
        # Check the summary file exists
        fname = cwd / "autonomy-summary-999.json"
        assert fname.exists()


    def test_gh_automerge_retries(monkeypatch):
        state = {"calls": 0}

        class FakeResp:
            def __init__(self, code=500):
                self.status_code = code
                self.text = ""

        def fake_put(url, headers=None, json=None):
            state["calls"] += 1
            if state["calls"] == 1:
                return FakeResp(500)
            return FakeResp(200)

        with patch("scripts.autonomy_check.requests.put", fake_put):
            ok = gh_automerge("owner", "repo", 12, "token")
            assert ok is True
