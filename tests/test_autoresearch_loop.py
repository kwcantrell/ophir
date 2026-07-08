"""Offline unit tests for the torch-free autoresearch loop runner."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_loop", REPO_ROOT / "autoresearch" / "loop.py"
)
assert _SPEC is not None and _SPEC.loader is not None
loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loop)


class TestDecide:
    def test_first_finite_result_becomes_baseline(self) -> None:
        assert loop.decide(0.01, None) is True

    def test_nan_candidate_is_rejected(self) -> None:
        assert loop.decide(float("nan"), None) is False
        assert loop.decide(float("nan"), 0.05) is False

    def test_none_candidate_is_rejected(self) -> None:
        assert loop.decide(None, 0.05) is False

    def test_must_beat_best_by_epsilon(self) -> None:
        # Clearly inside / clearly outside the band; no exact-FP-bound asserts.
        assert loop.decide(0.0649, 0.05, epsilon=0.02) is False
        assert loop.decide(0.0751, 0.05, epsilon=0.02) is True

    def test_default_epsilon_is_applied(self) -> None:
        assert loop.decide(0.055, 0.05) is False


class TestParsePorcelain:
    def test_modified_and_untracked_are_split(self) -> None:
        text = " M autoresearch/train_experiment.py\n?? autoresearch/runs/s1/.hypothesis\n"
        modified, untracked = loop.parse_porcelain(text)
        assert modified == ["autoresearch/train_experiment.py"]
        assert untracked == ["autoresearch/runs/s1/.hypothesis"]

    def test_empty_status_is_clean(self) -> None:
        assert loop.parse_porcelain("") == ([], [])

    def test_staged_and_renamed_count_as_modified(self) -> None:
        text = "M  a.py\nR  old.py -> new.py\n"
        modified, _ = loop.parse_porcelain(text)
        assert "a.py" in modified
        assert "new.py" in modified


class TestDiffIsValid:
    SESSION = "autoresearch/runs/s1"

    def test_exactly_the_mutable_file_is_valid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], [], self.SESSION) is True

    def test_no_edit_is_invalid(self) -> None:
        assert loop.diff_is_valid([], [], self.SESSION) is False

    def test_touching_other_tracked_files_is_invalid(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE, "src/ophir/safety.py"], [], self.SESSION)
            is False
        )

    def test_untracked_inside_session_dir_is_allowed(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE], [f"{self.SESSION}/.hypothesis"], self.SESSION)
            is True
        )

    def test_untracked_outside_session_dir_is_invalid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], ["evil.py"], self.SESSION) is False


class TestValidateExperimentSource:
    def test_valid_source_passes(self) -> None:
        text = f"import os\n{loop.SEALED_IMPORT_LINE}\nx = 128\nhidden = 2048\n"
        assert loop.validate_experiment_source(text) is None

    def test_missing_sealed_import_is_rejected(self) -> None:
        reason = loop.validate_experiment_source("x = 1\n")
        assert reason is not None and "sealed import" in reason

    def test_year_literal_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nval_max_year = 2025\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "year literal" in reason

    def test_non_year_numbers_are_fine(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nemb = 2048\nlr = 0.0002\nn = 10000\n"
        assert loop.validate_experiment_source(text) is None


class TestMetricsAndResults:
    def test_parse_metrics_reads_named_keys_only(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": 0.061, "h1": 0.09, "h5": 0.05, "n": 1000, "junk": 9}')
        metrics = loop.parse_metrics(str(p))
        assert metrics["rank_ic_near"] == 0.061
        assert "junk" not in metrics

    def test_parse_metrics_tolerates_nan(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": NaN}')
        assert math.isnan(loop.parse_metrics(str(p))["rank_ic_near"])

    def test_result_row_is_tab_separated_and_sanitized(self) -> None:
        row = loop.format_result_row(
            iteration=3,
            utc="2026-07-07T05:00:00Z",
            hypothesis="try\tranking\nloss",
            status="keep",
            rank_ic_near=0.061,
            h1=0.09,
            h5=0.05,
            wall_s=412.0,
            commit="abc1234",
        )
        cells = row.split("\t")
        assert len(cells) == len(loop.RESULTS_HEADER.split("\t"))
        assert cells[2] == "try ranking loss"
        assert cells[3] == "keep"

    def test_append_result_writes_header_once(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        loop.append_result(str(tsv), "1\trow")
        loop.append_result(str(tsv), "2\trow")
        lines = tsv.read_text().splitlines()
        assert lines[0] == loop.RESULTS_HEADER
        assert lines[1:] == ["1\trow", "2\trow"]


class FakeRunner:
    """Scripted subprocess stand-in: maps a command marker to (rc, output)."""

    def __init__(self, script: dict[str, tuple[int, str]]) -> None:
        self.script = script
        self.calls: list[list[str]] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: str,
        timeout: float | None = None,
        input_text: str | None = None,
    ) -> tuple[int, str]:
        self.calls.append(cmd)
        for marker, result in self.script.items():
            if marker in " ".join(cmd):
                return result
        return (0, "")

    def commands(self, marker: str) -> list[list[str]]:
        return [c for c in self.calls if marker in " ".join(c)]


BASE_SHA = "base0000"


def _make_session(tmp_path: Path, iter_name: str, metrics: str) -> str:
    session_dir = tmp_path / "runs" / "s1"
    iter_dir = session_dir / iter_name
    iter_dir.mkdir(parents=True)
    (iter_dir / "metrics.json").write_text(metrics)
    (iter_dir / "best-step=1.ckpt").write_text("stub")
    (session_dir / ".hypothesis").write_text("wider near-band loss weighting")
    return str(session_dir)


def _experiment_file_ok(monkeypatch, tmp_path: Path) -> None:
    exp = tmp_path / "train_experiment.py"
    exp.write_text(f"{loop.SEALED_IMPORT_LINE}\n")
    monkeypatch.setattr(loop, "MUTABLE_PATH", str(exp))


class TestRunIteration:
    def _propose_runner(self, metrics_ok: bool = True) -> FakeRunner:
        return FakeRunner({"status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n")})

    def test_keep_flow_commits_and_never_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner()
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.rank_ic_near == 0.08
        commit_cmds = runner.commands("git commit")
        assert commit_cmds and "--no-verify" in commit_cmds[0]
        assert not runner.commands("reset --hard")

    def test_discard_flow_resets_to_base_sha(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.031}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner()
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "discard"
        resets = runner.commands("reset --hard")
        assert resets and resets[0][-1] == BASE_SHA

    def test_invalid_diff_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"status --porcelain": (0, " M src/ophir/safety.py\n")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")
        assert runner.commands("reset --hard")

    def test_failed_commit_is_invalid_and_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "git commit": (1, "hook rejected"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_proposer_failure_is_its_own_status(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"claude": (1, "not logged in")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "proposer-fail"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_train_timeout_is_crash_and_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "train_experiment.py": (-1, "TIMEOUT"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "crash"
        assert runner.commands("reset --hard")

    def test_baseline_iteration_skips_proposal_and_never_commits(
        self, tmp_path, monkeypatch
    ) -> None:
        session_dir = _make_session(tmp_path, "iter-000", '{"rank_ic_near": 0.06}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({})
        result = loop.run_iteration(
            0, session_dir, None, BASE_SHA, propose=False, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.hypothesis == "baseline"
        assert not runner.commands("claude")
        assert not runner.commands("commit")
        assert not runner.commands("reset --hard")


class TestPromptAndPins:
    def test_prompt_carries_contract_and_context(self) -> None:
        prompt = loop.build_prompt("PROGRAM", "iter\t...", "abc fix loss", "/tmp/s/.hypothesis")
        assert "PROGRAM" in prompt
        assert loop.MUTABLE_FILE in prompt
        assert ".hypothesis" in prompt
        assert "one" in prompt.lower()

    def test_pins_detect_tampering(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "eval_harness.py"
        target.write_text("original")
        monkeypatch.setattr(loop, "PINNED_FILES", (str(target),))
        pins = loop.pin_hashes()
        assert loop.check_pins(pins) == []
        target.write_text("tampered")
        assert loop.check_pins(pins) == [str(target)]
