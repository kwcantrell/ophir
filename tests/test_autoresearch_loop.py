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
