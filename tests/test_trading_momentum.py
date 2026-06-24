from ophir.trading.momentum import momentum_score


def _series(n: int, drift: float, base: float = 100.0, noise: float = 0.003) -> list[float]:
    """Deterministic price path with ``drift`` mean daily return plus alternating
    noise, so daily returns *vary* (nonzero variance — a pure geometric path has
    constant returns and an undefined information ratio)."""
    closes = [base]
    for i in range(1, n):
        ret = drift + (noise if i % 2 == 0 else -noise)
        closes.append(closes[-1] * (1.0 + ret))
    return closes


def test_momentum_score_rising_is_positive() -> None:
    assert (momentum_score(_series(80, 0.01), lookback=63, skip=5) or 0.0) > 0.0


def test_momentum_score_falling_is_negative() -> None:
    assert (momentum_score(_series(80, -0.01), lookback=63, skip=5) or 0.0) < 0.0


def test_momentum_score_constant_series_is_none() -> None:
    # Zero variance -> undefined information ratio.
    assert momentum_score([100.0] * 80, lookback=63, skip=5) is None


def test_momentum_score_insufficient_history_is_none() -> None:
    # Need lookback + skip + 1 = 69 closes; 68 is too few.
    assert momentum_score(_series(68, 0.01), lookback=63, skip=5) is None


def test_momentum_score_skip_excludes_recent_spike() -> None:
    base = _series(80, 0.01)
    spiked = [*base[:-3], base[-3] * 0.5, base[-2] * 0.5, base[-1] * 0.5]
    # The crash lands inside the skipped 5-bar tail, so the score is unchanged.
    assert momentum_score(spiked, lookback=63, skip=5) == momentum_score(base, lookback=63, skip=5)


def test_momentum_score_lookback_below_two_is_none() -> None:
    # A window of <2 daily returns has undefined sample std -> degrade to None.
    assert momentum_score(_series(80, 0.01), lookback=1, skip=5) is None
