# Forecast-Ceiling Confirmation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only harness that confirms the E3 near-offset cross-sectional skill multi-seed, with a per-offset permutation null, then run the GPU seeds that feed it.

**Architecture:** Three new pure functions in `ophir.ceiling` — a per-offset within-day permutation null, a per-run/multi-snapshot `val_rank_ic_h*` aggregator, and a report assembler that joins them into a per-offset verdict — plus a thin `scripts/` wrapper. The null is computed *target-vs-shuffled-self* so it needs no model. A small supporting change promotes `_trading_day_offsets` to public so the harvest can reuse it.

**Tech Stack:** Python 3.10+, PyTorch (CPU tensors), pandas, NumPy, pytest. Reuses `ophir.ceiling` primitives (`shuffle_within_day`, `cross_sectional_ic`, `aggregate_ic`, `mde_for_group_difference`, `_pick_column`) and `ophir.evaluate` rank-IC math.

## Global Constraints

- mypy `strict = True`, targets **Python 3.10** (lowest runtime). Keep `src/ophir` fully typed.
- ruff targets **3.12**. Do not change either floor to match the other.
- pytest runs `filterwarnings = error`; tests must be **offline + CPU-only**, using `tmp_path` and seeded deterministic fixtures. Never touch the network, CUDA, or `.ophir/`.
- **NumPy-style docstrings** throughout `src/ophir`; match existing density.
- Imports: `known-first-party = ["ophir"]` (ruff/isort ordering).
- Reuse the **production rank-IC math** (`cross_sectional_ic` → `dedupe_by_ticker_date` + `rank_ic`) so the offline null and the live `val_rank_ic` agree.
- Branch: `forecast-ceiling-confirm-harness`. Stage only files this plan touches; leave the pre-existing dirty files (`.claude/settings.json`, `.gitignore`, `docs/forecast-ceiling-fix-context.md`, `docs/rezero-init-sweep-runbook.md`, `.graphifyignore`) unstaged.
- Verification gate before any "done" claim: `uv run pytest tests/test_ceiling.py -q`, `uv run ruff check . && uv run ruff format --check .`, `uv run mypy src/ophir`.

---

### Task 1: Promote `_trading_day_offsets` to public `trading_day_offsets`

**Files:**
- Modify: `src/ophir/training_models.py:24` (def), `:465` (caller)
- Modify: `tests/test_training_models.py:8` (import), `:191-196` (test)

**Interfaces:**
- Consumes: nothing.
- Produces: `trading_day_offsets(trade_occured: torch.Tensor) -> torch.Tensor` — public, pure CPU; 1-based trading-day rank (`cumsum` over the trade mask). Reused by Task 4/5's harvest.

- [ ] **Step 1: Update the existing test to import/use the public name**

In `tests/test_training_models.py`, change the import (line 8) from `_trading_day_offsets` to `trading_day_offsets`, and the call site in `test_trading_day_offsets_counts_only_trading_days` (line ~196):

```python
from ophir.training_models import (
    trading_day_offsets,
)
...
def test_trading_day_offsets_counts_only_trading_days() -> None:
    mask = torch.tensor([[1, 0, 1, 1]])
    offsets = trading_day_offsets(mask)
    assert offsets.tolist() == [[1, 1, 2, 3]]
```

(Keep the rest of that test body as-is; only the symbol name changes.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_training_models.py::test_trading_day_offsets_counts_only_trading_days -q`
Expected: FAIL — `ImportError: cannot import name 'trading_day_offsets'`.

- [ ] **Step 3: Rename the function and update the internal caller**

In `src/ophir/training_models.py`, rename `def _trading_day_offsets(` (line 24) to `def trading_day_offsets(`, leaving the docstring/body unchanged. Update the call at line ~465 from `offsets = _trading_day_offsets(mask)` to `offsets = trading_day_offsets(mask)`. Update the docstring cross-reference in `signal_decay_curve`/elsewhere only if it names `_trading_day_offsets` (search and fix).

```python
def trading_day_offsets(trade_occured: torch.Tensor) -> torch.Tensor:
    """1-based trading-day rank of each position within its row.
    ...unchanged docstring...
    """
    return trade_occured.long().cumsum(dim=1)
```

- [ ] **Step 4: Verify the test passes and nothing else broke**

Run: `uv run pytest tests/test_training_models.py -q && uv run mypy src/ophir`
Expected: PASS; mypy clean. (Confirm no remaining `_trading_day_offsets` references: `grep -rn "_trading_day_offsets" src/ tests/` returns nothing.)

- [ ] **Step 5: Commit**

```bash
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Promote _trading_day_offsets to public trading_day_offsets"
```

---

### Task 2: Per-offset permutation null (`NullBand` + `per_offset_shuffle_null`)

**Files:**
- Modify: `src/ophir/ceiling.py` (add after `shuffle_within_day`)
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `shuffle_within_day`, `cross_sectional_ic` (existing in `ceiling.py`).
- Produces:
  - `NullBand` frozen dataclass: `mean, std, p05, p95 (float); n_perms, n_rows (int)`.
  - `per_offset_shuffle_null(target, ids, dates, offsets, buckets, *, n_perms, generator) -> dict[str, NullBand]` — keys `"h{offset}"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ceiling.py` (extend the existing `from ophir.ceiling import (...)` block with `NullBand` and `per_offset_shuffle_null`):

```python
def test_per_offset_shuffle_null_brackets_zero_and_widens_when_thin() -> None:
    n_days, n_names = 20, 8
    tg = torch.Generator().manual_seed(1)
    target = torch.randn(n_days * n_names, generator=tg)
    ids = torch.arange(n_names).repeat(n_days)
    dates = torch.arange(n_days).repeat_interleave(n_names)
    offsets = torch.ones(n_days * n_names, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    bands = per_offset_shuffle_null(
        target, ids, dates, offsets, [1, 2], n_perms=200, generator=g
    )
    assert abs(bands["h1"].mean) < bands["h1"].std  # null centered on ~0
    assert bands["h1"].p05 < 0.0 < bands["h1"].p95
    assert bands["h1"].n_rows == n_days * n_names
    assert bands["h1"].n_perms == 200


def test_per_offset_shuffle_null_empty_bucket_is_nan() -> None:
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ids = torch.tensor([1, 2, 1, 2])
    dates = torch.tensor([1, 1, 2, 2])
    offsets = torch.tensor([1, 1, 1, 1])
    g = torch.Generator().manual_seed(0)
    bands = per_offset_shuffle_null(
        target, ids, dates, offsets, [1, 90], n_perms=50, generator=g
    )
    assert bands["h90"].n_rows == 0
    assert math.isnan(bands["h90"].mean) and math.isnan(bands["h90"].p95)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ceiling.py::test_per_offset_shuffle_null_brackets_zero_and_widens_when_thin -q`
Expected: FAIL — `ImportError: cannot import name 'NullBand'`.

- [ ] **Step 3: Implement `NullBand` + `per_offset_shuffle_null`**

In `src/ophir/ceiling.py`, add the dataclass next to the other `@dataclass(frozen=True)` blocks and the function after `shuffle_within_day`:

```python
@dataclass(frozen=True)
class NullBand:
    """Within-day permutation-null IC distribution for one offset bucket.

    Attributes
    ----------
    mean, std : float
        Mean and sample std (``ddof=1``) of the null cross-sectional IC over
        ``n_perms`` within-day target shuffles. ``mean`` is expected near 0.
    p05, p95 : float
        5th / 95th percentiles of the null IC distribution — the chance band a
        real per-offset IC must clear to be called signal.
    n_perms, n_rows : int
        Permutation count and the number of rows in this offset bucket.
    """

    mean: float
    std: float
    p05: float
    p95: float
    n_perms: int
    n_rows: int


def per_offset_shuffle_null(
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    buckets: Sequence[int],
    *,
    n_perms: int,
    generator: torch.Generator,
) -> dict[str, NullBand]:
    """Per-offset within-day permutation null for cross-sectional rank-IC.

    For each bucket ``h`` the rows with ``offsets == h`` are isolated and the
    target is permuted within each day ``n_perms`` times (via
    :func:`shuffle_within_day`); each shuffle's cross-sectional IC (via
    :func:`cross_sectional_ic`, the production metric) forms the null. The band
    depends only on the per-day cross-section group sizes, not on the identity
    of the signal, so correlating the target against a within-day shuffle of
    itself yields the same band as shuffling against model predictions — no
    model is needed. Thinner near-offset cross-sections give wider bands.

    Parameters
    ----------
    target, ids, dates, offsets : torch.Tensor
        Equal-length 1-D tensors: target value, ticker id, integer date ordinal,
        and 1-based trading-day offset for each response observation.
    buckets : sequence of int
        Offsets to report; ``"h{offset}"`` keys mirror
        :func:`ophir.evaluate.rank_ic_by_offset`.
    n_perms : int
        Number of within-day shuffles.
    generator : torch.Generator
        Advanced in place; seed it for reproducibility.

    Returns
    -------
    dict[str, NullBand]
        One band per bucket; an empty bucket yields a ``nan`` band.
    """
    out: dict[str, NullBand] = {}
    for h in buckets:
        key = f"h{int(h)}"
        sel = offsets == int(h)
        n_rows = int(sel.sum())
        if n_rows == 0:
            out[key] = NullBand(
                float("nan"), float("nan"), float("nan"), float("nan"), n_perms, 0
            )
            continue
        t_h, i_h, d_h = target[sel], ids[sel], dates[sel]
        ics = [
            cross_sectional_ic(
                t_h, shuffle_within_day(t_h, d_h, generator=generator), i_h, d_h
            )["ic_mean"]
            for _ in range(n_perms)
        ]
        finite = np.asarray(ics, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            out[key] = NullBand(
                float("nan"), float("nan"), float("nan"), float("nan"), n_perms, n_rows
            )
            continue
        out[key] = NullBand(
            mean=float(finite.mean()),
            std=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            p05=float(np.percentile(finite, 5)),
            p95=float(np.percentile(finite, 95)),
            n_perms=n_perms,
            n_rows=n_rows,
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ceiling.py -k per_offset_shuffle_null -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add per-offset within-day permutation null to ophir.ceiling"
```

---

### Task 3: Multi-snapshot per-offset aggregation (`OffsetRunIC` + `run_offset_ic`)

**Files:**
- Modify: `src/ophir/ceiling.py` (add after `run_ic_summary`)
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `_pick_column` (existing); pandas.
- Produces:
  - `OffsetRunIC` frozen dataclass: `snapshot_mean, peak (float); n_snapshots (int)`.
  - `run_offset_ic(metrics_csv, buckets, *, burn_in_steps=0) -> dict[str, OffsetRunIC]` — keys `"h{offset}"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ceiling.py` (extend the import block with `OffsetRunIC, run_offset_ic`):

```python
def test_run_offset_ic_means_snapshots_and_reports_peak(tmp_path: Path) -> None:
    rows = [
        {"step": 100, "val_rank_ic_h1": 0.02},
        {"step": 200, "val_rank_ic_h1": 0.10},
        {"step": 300, "val_rank_ic_h1": 0.06},
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1], burn_in_steps=0)
    assert out["h1"].n_snapshots == 3
    assert out["h1"].peak == pytest.approx(0.10)
    assert out["h1"].snapshot_mean == pytest.approx(0.06)


def test_run_offset_ic_burn_in_excludes_early_steps(tmp_path: Path) -> None:
    rows = [
        {"step": 50, "val_rank_ic_h1": 0.0},
        {"step": 500, "val_rank_ic_h1": 0.08},
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1], burn_in_steps=100)
    assert out["h1"].n_snapshots == 1
    assert out["h1"].snapshot_mean == pytest.approx(0.08)


def test_run_offset_ic_missing_bucket_is_nan(tmp_path: Path) -> None:
    rows = [{"step": 100, "val_rank_ic_h1": 0.05, "val_rank_ic_h90": float("nan")}]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1, 90])
    assert out["h1"].snapshot_mean == pytest.approx(0.05)
    assert math.isnan(out["h90"].snapshot_mean) and out["h90"].n_snapshots == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ceiling.py -k run_offset_ic -q`
Expected: FAIL — `ImportError: cannot import name 'OffsetRunIC'`.

- [ ] **Step 3: Implement `OffsetRunIC` + `run_offset_ic`**

In `src/ophir/ceiling.py`:

```python
@dataclass(frozen=True)
class OffsetRunIC:
    """Per-offset ``val_rank_ic_h*`` summary for one training run.

    Attributes
    ----------
    snapshot_mean : float
        Mean ``val_rank_ic_h{offset}`` over the run's non-burn-in validation
        snapshots — the denoised headline estimate.
    peak : float
        Max over those snapshots (the ceiling; also flags the E0 mid-run droop).
    n_snapshots : int
        Number of snapshots averaged.
    """

    snapshot_mean: float
    peak: float
    n_snapshots: int


def run_offset_ic(
    metrics_csv: str | Path,
    buckets: Sequence[int],
    *,
    burn_in_steps: int = 0,
) -> dict[str, OffsetRunIC]:
    """Summarise per-offset ``val_rank_ic_h*`` from a run's ``metrics.csv``.

    Averages each ``val_rank_ic_h{offset}`` column over validation snapshots
    (dropping NaN rows and rows with ``step < burn_in_steps``) and records the
    peak. A bucket whose column is absent or all-NaN yields a ``nan`` summary
    with ``n_snapshots == 0`` rather than raising.

    Parameters
    ----------
    metrics_csv : str or Path
        Lightning CSVLogger ``metrics.csv`` from an ``--log-offset-ic`` run.
    buckets : sequence of int
        Offsets to summarise (``_OFFSET_BUCKETS`` in production).
    burn_in_steps : int, optional
        Exclude validation rows logged before this global step (default 0).

    Returns
    -------
    dict[str, OffsetRunIC]
        One summary per bucket, keyed ``"h{offset}"``.
    """
    df = pd.read_csv(metrics_csv)
    step_col = _pick_column(df, ("step",))
    out: dict[str, OffsetRunIC] = {}
    for h in buckets:
        key = f"h{int(h)}"
        try:
            col = _pick_column(df, (f"val_rank_ic_{key}", f"val_rank_ic_{key}_epoch"))
        except KeyError:
            out[key] = OffsetRunIC(float("nan"), float("nan"), 0)
            continue
        sub = df.dropna(subset=[col])
        sub = sub[sub[step_col] >= burn_in_steps]
        if sub.empty:
            out[key] = OffsetRunIC(float("nan"), float("nan"), 0)
            continue
        vals = sub[col].to_numpy(dtype=float)
        out[key] = OffsetRunIC(
            snapshot_mean=float(vals.mean()),
            peak=float(vals.max()),
            n_snapshots=int(vals.size),
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ceiling.py -k run_offset_ic -q`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add per-offset multi-snapshot run aggregation to ophir.ceiling"
```

---

### Task 4: Report assembler (`OffsetVerdict`, `confirm_offset_skill`, `format_verdict_table`)

**Files:**
- Modify: `src/ophir/ceiling.py` (add after Task 3 code)
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `per_offset_shuffle_null`, `NullBand` (Task 2); `run_offset_ic`, `OffsetRunIC` (Task 3); `aggregate_ic` (existing).
- Produces:
  - `OffsetVerdict` frozen dataclass: `offset (int); seed_mean, seed_std (float); n_seeds (int); peak (float); null (NullBand); clears_null (bool)`.
  - `confirm_offset_skill(metrics_csvs, harvest, buckets, *, n_perms=500, burn_in_steps=0, seed=0) -> list[OffsetVerdict]` where `harvest = (target, ids, dates, offsets)`.
  - `format_verdict_table(verdicts) -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ceiling.py` (extend imports with `OffsetVerdict, confirm_offset_skill, format_verdict_table`):

```python
def _planted_harvest() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 20 days x 8 names in EACH of buckets h1 and h2, so both get a real null
    # band (~p95 0.14); wide enough that 0.29 clears and 0.05 does not.
    n_days, n_names = 20, 8
    n = n_days * n_names
    tg = torch.Generator().manual_seed(7)
    base_ids = torch.arange(n_names).repeat(n_days)
    base_dates = torch.arange(n_days).repeat_interleave(n_names)
    target = torch.randn(2 * n, generator=tg)
    ids = torch.cat([base_ids, base_ids])
    dates = torch.cat([base_dates, base_dates])
    offsets = torch.cat(
        [torch.ones(n, dtype=torch.long), torch.full((n,), 2, dtype=torch.long)]
    )
    return target, ids, dates, offsets


def _write_offset_metrics(path: Path, h1: float, h2: float) -> Path:
    pd.DataFrame(
        [
            {"step": 200, "val_rank_ic_h1": h1, "val_rank_ic_h2": h2},
            {"step": 400, "val_rank_ic_h1": h1, "val_rank_ic_h2": h2},
        ]
    ).to_csv(path, index=False)
    return path


def test_confirm_offset_skill_flags_signal_vs_noise(tmp_path: Path) -> None:
    csvs = [
        _write_offset_metrics(tmp_path / "s0.csv", h1=0.30, h2=0.04),
        _write_offset_metrics(tmp_path / "s1.csv", h1=0.28, h2=0.06),
    ]
    verdicts = confirm_offset_skill(
        csvs, _planted_harvest(), [1, 2], n_perms=200, seed=0
    )
    by_offset = {v.offset: v for v in verdicts}
    assert by_offset[1].n_seeds == 2
    assert by_offset[1].seed_mean == pytest.approx(0.29, abs=1e-6)
    assert by_offset[1].clears_null is True   # 0.29 > h1 null p95
    assert by_offset[2].clears_null is False  # ~0.05 inside its null band


def test_format_verdict_table_has_header_and_row_per_offset(tmp_path: Path) -> None:
    csvs = [_write_offset_metrics(tmp_path / "s0.csv", h1=0.30, h2=0.04)]
    verdicts = confirm_offset_skill(csvs, _planted_harvest(), [1, 2], n_perms=50)
    table = format_verdict_table(verdicts)
    lines = table.splitlines()
    assert "offset" in lines[0] and "clears" in lines[0]
    assert len(lines) == 3  # header + 2 offsets
    assert "yes" in lines[1]  # offset 1 clears
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ceiling.py -k "confirm_offset_skill or format_verdict_table" -q`
Expected: FAIL — `ImportError: cannot import name 'OffsetVerdict'`.

- [ ] **Step 3: Implement the assembler**

In `src/ophir/ceiling.py`:

```python
@dataclass(frozen=True)
class OffsetVerdict:
    """Per-offset confirmation verdict joining seed aggregate to the null band.

    Attributes
    ----------
    offset : int
        Trading-day-lead bucket.
    seed_mean, seed_std : float
        Cross-seed mean and sample std of each run's ``snapshot_mean``.
    n_seeds : int
        Number of runs (seeds) contributing a finite value.
    peak : float
        Cross-seed mean of each run's per-offset peak (diagnostic).
    null : NullBand
        The bucket's within-day permutation null.
    clears_null : bool
        ``True`` iff ``seed_mean`` exceeds the null 95th percentile.
    """

    offset: int
    seed_mean: float
    seed_std: float
    n_seeds: int
    peak: float
    null: NullBand
    clears_null: bool


def confirm_offset_skill(
    metrics_csvs: Sequence[str | Path],
    harvest: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    buckets: Sequence[int],
    *,
    n_perms: int = 500,
    burn_in_steps: int = 0,
    seed: int = 0,
) -> list[OffsetVerdict]:
    """Confirm per-offset skill across seeds against a per-offset null.

    Computes the within-day permutation null once from ``harvest`` (model-free),
    aggregates each run's ``snapshot_mean`` across seeds via
    :func:`aggregate_ic`, and flags ``clears_null`` where the seed-mean exceeds
    the null 95th percentile.

    Parameters
    ----------
    metrics_csvs : sequence of str or Path
        One ``metrics.csv`` per seed run.
    harvest : tuple of torch.Tensor
        ``(target, ids, dates, offsets)`` from the CPU validation harvest.
    buckets : sequence of int
        Offsets to report.
    n_perms : int, optional
        Null permutations (default 500).
    burn_in_steps : int, optional
        Snapshot burn-in passed to :func:`run_offset_ic` (default 0).
    seed : int, optional
        Seeds the null's ``torch.Generator`` (default 0).

    Returns
    -------
    list[OffsetVerdict]
        One verdict per bucket, in ``buckets`` order.
    """
    target, ids, dates, offsets = harvest
    generator = torch.Generator().manual_seed(seed)
    null = per_offset_shuffle_null(
        target, ids, dates, offsets, buckets, n_perms=n_perms, generator=generator
    )
    per_run = [
        run_offset_ic(csv, buckets, burn_in_steps=burn_in_steps) for csv in metrics_csvs
    ]
    verdicts: list[OffsetVerdict] = []
    for h in buckets:
        key = f"h{int(h)}"
        means = [r[key].snapshot_mean for r in per_run if r[key].snapshot_mean == r[key].snapshot_mean]
        peaks = [r[key].peak for r in per_run if r[key].peak == r[key].peak]
        band = null[key]
        if means:
            agg = aggregate_ic(means)
            seed_mean, seed_std, n_seeds = agg.mean, agg.std, agg.n
        else:
            seed_mean, seed_std, n_seeds = float("nan"), float("nan"), 0
        peak = float(sum(peaks) / len(peaks)) if peaks else float("nan")
        clears = bool(
            seed_mean == seed_mean and band.p95 == band.p95 and seed_mean > band.p95
        )
        verdicts.append(
            OffsetVerdict(int(h), seed_mean, seed_std, n_seeds, peak, band, clears)
        )
    return verdicts


def format_verdict_table(verdicts: Sequence[OffsetVerdict]) -> str:
    """Render :func:`confirm_offset_skill` verdicts as a fixed-width table."""
    header = (
        f"{'offset':>6} {'seed_mean':>10} {'seed_std':>9} {'n':>3} "
        f"{'peak':>8} {'null_p95':>9} {'clears':>7}"
    )
    lines = [header]
    for v in verdicts:
        lines.append(
            f"{v.offset:>6} {v.seed_mean:>10.4f} {v.seed_std:>9.4f} {v.n_seeds:>3} "
            f"{v.peak:>8.4f} {v.null.p95:>9.4f} {'yes' if v.clears_null else 'no':>7}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ceiling.py -q && uv run mypy src/ophir`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add confirm_offset_skill report assembler to ophir.ceiling"
```

---

### Task 5: Thin CLI script `scripts/confirm_offset_skill.py`

**Files:**
- Create: `scripts/confirm_offset_skill.py`
- Test: `tests/test_ceiling.py` (smoke test of the load+assemble path it wraps)

**Interfaces:**
- Consumes: `confirm_offset_skill`, `format_verdict_table` (Task 4); `_OFFSET_BUCKETS` (`ophir.training_models`).
- Produces: a runnable script; no importable logic beyond what Task 4 already tests.

- [ ] **Step 1: Write the failing test** (round-trips a saved harvest `.pt` through the script's load helper)

Add to `tests/test_ceiling.py`:

```python
def test_load_harvest_round_trips(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "confirm_offset_skill_script",
        Path(__file__).resolve().parents[1] / "scripts" / "confirm_offset_skill.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    harvest = {
        "target": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "ids": torch.tensor([1, 2, 1, 2]),
        "dates": torch.tensor([1, 1, 2, 2]),
        "offsets": torch.tensor([1, 1, 1, 1]),
    }
    path = tmp_path / "harvest.pt"
    torch.save(harvest, path)
    target, ids, dates, offsets = mod.load_harvest(path)
    assert target.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert offsets.tolist() == [1, 1, 1, 1]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ceiling.py::test_load_harvest_round_trips -q`
Expected: FAIL — `FileNotFoundError` / cannot load `scripts/confirm_offset_skill.py`.

- [ ] **Step 3: Create the script**

`scripts/confirm_offset_skill.py`:

```python
"""Confirm per-offset cross-sectional skill multi-seed against a per-offset null.

Joins the logged ``val_rank_ic_h*`` from one or more ``--log-offset-ic`` training
runs (averaged across validation snapshots and seeds) to a within-day permutation
null computed from a CPU validation harvest, and prints a per-offset verdict
table (see ``ophir.ceiling.confirm_offset_skill``).

The harvest is a ``torch.save`` dict with tensors ``target``, ``ids``, ``dates``,
``offsets`` (offsets via ``ophir.training_models.trading_day_offsets``). Run with::

    uv run python scripts/confirm_offset_skill.py \
        --harvest harvest.pt \
        --metrics .ophir/.../version_0/metrics.csv .ophir/.../version_1/metrics.csv

CPU-only; no model or CUDA required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ophir.ceiling import confirm_offset_skill, format_verdict_table
from ophir.training_models import _OFFSET_BUCKETS


def load_harvest(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a saved harvest dict into a ``(target, ids, dates, offsets)`` tuple."""
    blob = torch.load(path, map_location="cpu")
    return blob["target"], blob["ids"], blob["dates"], blob["offsets"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, nargs="+", type=Path)
    parser.add_argument("--n-perms", type=int, default=500)
    parser.add_argument("--burn-in-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verdicts = confirm_offset_skill(
        args.metrics,
        load_harvest(args.harvest),
        list(_OFFSET_BUCKETS),
        n_perms=args.n_perms,
        burn_in_steps=args.burn_in_steps,
        seed=args.seed,
    )
    print(format_verdict_table(verdicts))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass + lint/type the script**

Run: `uv run pytest tests/test_ceiling.py::test_load_harvest_round_trips -q && uv run ruff check scripts/confirm_offset_skill.py`
Expected: PASS; ruff clean. (mypy targets `src/ophir` only; the script is checked by ruff.)

- [ ] **Step 5: Commit**

```bash
git add scripts/confirm_offset_skill.py tests/test_ceiling.py
git commit -m "Add scripts/confirm_offset_skill.py harness wrapper"
```

---

### Task 6: Changelog + full verification gate

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:** none.

- [ ] **Step 1: Add a changelog entry**

Under `[Unreleased]` in `CHANGELOG.md`, add (match surrounding bullet style):

```markdown
- Add `ophir.ceiling` confirmation harness for the forecast-ceiling fix:
  `per_offset_shuffle_null` (per-offset within-day permutation null),
  `run_offset_ic` (multi-snapshot `val_rank_ic_h*` aggregation), and
  `confirm_offset_skill` + `scripts/confirm_offset_skill.py` (multi-seed
  per-offset verdict table). Promote `_trading_day_offsets` to public
  `trading_day_offsets`.
```

- [ ] **Step 2: Run the full project gate**

Run:
```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src/ophir
```
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "Note forecast-ceiling confirmation harness in changelog"
```

---

### Task 7: GPU confirmation run (operational — not TDD)

> Run after Tasks 1–6 land. Requires the RTX 3090 via `uv run`; **free the `llama-server` holding ~18.5 GB first**, or pass a smaller `--batch-size`.

- [ ] **Step 1: Multi-seed proxy training with offset-IC logging**

For `seed in 0 1 2`:
```bash
uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
    --max-steps 10000 --seed "$seed" --val-identity --log-offset-ic
```
(~6 min each. `version_260` may be reused as seed 0 if its config matches; otherwise run all three fresh.) Note each run's `metrics.csv` path under `.ophir/.../version_*/`.

- [ ] **Step 2: Harvest the validation cross-section (CPU, model-free)**

Using the E3 Step-A snippet (`docs/superpowers/plans/2026-06-21-forecast-horizon-diagnostic.md`, Task 5): build the val loader via `build_split_handlers` / `build_dataloader(..., return_identity=True)`, read `target_r_close` / `stock_id` / `date_ordinal` off `OHLCMulitClassPredictorInput`, derive offsets via `trading_day_offsets(trade_occured)`, and `torch.save({"target":..., "ids":..., "dates":..., "offsets":...}, "harvest.pt")`.

- [ ] **Step 3: Run the harness and record the result**

```bash
uv run python scripts/confirm_offset_skill.py \
    --harvest harvest.pt \
    --metrics .ophir/.../version_0/metrics.csv \
              .ophir/.../version_1/metrics.csv \
              .ophir/.../version_2/metrics.csv
```
Record the table in `docs/forecast-ceiling-results.md` (new E-section). If the near-offset magnitudes shift materially from Step B's ~0.06–0.10, update the `forecast-ceiling-investigation` memory. The fix (design-space item 2) is authorized iff near offsets (1–5) clear their p95 null across seeds and far offsets (≥40) sit inside theirs.

---

## Self-Review

**Spec coverage:**
- Component 1 (per-offset null) → Task 2. ✓
- Component 2 (multi-seed/snapshot aggregation) → Task 3 (`run_offset_ic`) + Task 4 (cross-seed via `aggregate_ic`). ✓
- Component 3 (report assembler + script) → Task 4 (`confirm_offset_skill`, `format_verdict_table`) + Task 5 (script). ✓
- Supporting change (promote `_trading_day_offsets`) → Task 1. ✓
- GPU execution plan → Task 7. ✓
- Testing section (null brackets 0, significance discrimination, denoising, aggregation, degeneracy, `_epoch` tolerance) → Tasks 2–5 tests. `_epoch` tolerance is covered by reusing `_pick_column` with both spellings in `run_offset_ic` (Task 3 Step 3). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `NullBand`/`OffsetRunIC`/`OffsetVerdict` field names and the `"h{offset}"` key convention are used identically across Tasks 2–5; `confirm_offset_skill` signature matches the spec and the script's call site; `harvest` tuple order `(target, ids, dates, offsets)` is consistent between `confirm_offset_skill`, `load_harvest`, and the harvest save in Task 7. ✓
