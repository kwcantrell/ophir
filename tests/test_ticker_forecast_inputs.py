"""Tests for the live-inference window builder."""

from ophir.ticker import build_latest_inputs


def test_build_latest_inputs_known_skips_unknown_and_short(parquet_dir) -> None:
    # parquet_dir (conftest): AAA = full history, BBB = ~5-day span (too short
    # to form a seq_len window), CCC = constant volume. Plus a decoy _logs dir.
    base_path, _ = parquet_dir
    out = build_latest_inputs(["AAA", "BBB", "ZZZ"], seq_len=15, base_path=base_path)

    assert "AAA" in out  # full history -> one most-recent window
    assert "BBB" not in out  # ~5 days < seq_len=15 -> no window
    assert "ZZZ" not in out  # unknown symbol -> skipped, no raise

    inp = out["AAA"]
    assert {"feature_input", "targets", "trade_occured", "response_size"} <= set(inp)
    assert inp["feature_input"].shape[0] == 15
    assert int(inp["response_size"].squeeze()) == 1


def test_build_latest_inputs_empty_for_all_unknown(parquet_dir) -> None:
    base_path, _ = parquet_dir
    assert build_latest_inputs(["NOPE", "ALSONOPE"], seq_len=15, base_path=base_path) == {}
