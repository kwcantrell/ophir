"""End-to-end leakage check on real data through the actual model forward.

Unlike ``test_models_leakage.py`` (which pins the masking helper on synthetic
CPU tensors), this runs a *real* S&P 500 window through the full GPU
flex-attention forward and proves the model's output does not depend on the
response-block inputs (which are the targets). A positive control perturbs the
prefix instead and asserts the output *does* change, so the test cannot pass
vacuously.

Requires CUDA, the per-stock parquet tree, and a base checkpoint, so it is
skipped wherever those are absent (e.g. CI). Run locally with:

    uv run pytest tests/test_models_leakage_realdata.py -q -s
"""

import os

import pytest
import torch

from ophir.register import DATA_DIR, MODEL_DIR
from ophir.ticker import StockHanlder, extract_model_data

SEQ_LEN = 365
RESPONSE_SIZE = 90
SYMBOL = "AAPL"
BASE_PATH = os.path.join(DATA_DIR, "days", "stocks")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="model forward requires CUDA"),
    pytest.mark.skipif(
        not os.path.isdir(BASE_PATH), reason=f"per-stock parquet tree missing at {BASE_PATH}"
    ),
    pytest.mark.skipif(
        not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR),
        reason="no base checkpoint available",
    ),
]


@pytest.fixture(scope="module")
def real_window():
    """A real 365-day preprocessed feature window for ``SYMBOL`` (offline)."""
    handler = StockHanlder(
        seq_len=SEQ_LEN,
        base_path=BASE_PATH,
        return_stock_id=False,
        return_streamer=True,
        stock_splits=None,  # offline: no Yahoo/Wikipedia calls
        offset=RESPONSE_SIZE,
        min_volume=1000,
    )
    if SYMBOL not in handler.stocks:
        pytest.skip(f"{SYMBOL} not present in data")
    streamer = handler[SYMBOL]
    window = streamer.preprocessed_ohlc_df.iloc[-SEQ_LEN:]
    if len(window) < SEQ_LEN:
        pytest.skip(f"{SYMBOL} has only {len(window)}/{SEQ_LEN} days")
    return window


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import warnings

    import torch as _torch

    from ophir.register import BASE_NAME, MODEL_DIR, TIME_MODIFIER, _latest_base_ckpt
    from ophir.training_models import LightningOHLCPredictor

    # strict=False: checkpoints predate the new mask_token; random init is fine
    # because the leakage property we test is architectural, not weight-dependent.
    # Dropping the time_delta feature also shrank feature_mlp 13->12, so the saved
    # feature_mlp weights no longer fit -- strip them (same fresh-init rationale)
    # rather than fail to load. Re-train invalidates this caveat.
    name = (BASE_NAME + TIME_MODIFIER).split("{")[0]
    ckpt_path = os.path.join(MODEL_DIR, _latest_base_ckpt(filename=name))
    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if "feature_mlp" not in k}
    filtered = tmp_path_factory.mktemp("ckpt") / "no_feature_mlp.ckpt"
    _torch.save(ckpt, filtered)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return LightningOHLCPredictor.load_from_checkpoint(filtered, strict=False).cuda().eval()


def _forward(model, window):
    md = extract_model_data(window, response_size=RESPONSE_SIZE, return_date=True)
    with torch.no_grad():
        return model(md).model_output.detach().clone()


def _forward_perturbed(model, window, rows):
    """Run forward with ``feature_input`` rows ``rows`` replaced by large noise."""
    md = extract_model_data(window, response_size=RESPONSE_SIZE, return_date=True)
    torch.manual_seed(0)
    # feature_input is (seq_len, 12) here; the batch dim is added downstream.
    noise = torch.randn_like(md["feature_input"][rows, :]) * 1000.0
    md["feature_input"][rows, :] = noise
    with torch.no_grad():
        return model(md).model_output.detach().clone()


def test_output_shape(model, real_window):
    out = _forward(model, real_window)
    assert out.shape == (1, RESPONSE_SIZE, 3)


def test_no_leakage_from_response_block(model, real_window):
    """Corrupting the response-block inputs must not change the prediction."""
    baseline = _forward(model, real_window)
    perturbed = _forward_perturbed(model, real_window, slice(-RESPONSE_SIZE, None))
    assert torch.equal(baseline, perturbed), (
        "model output changed when response-block inputs were corrupted -> leakage"
    )


def test_prefix_perturbation_changes_output(model, real_window):
    """Positive control: the model must actually consume the prefix context."""
    baseline = _forward(model, real_window)
    perturbed = _forward_perturbed(model, real_window, slice(0, SEQ_LEN - RESPONSE_SIZE))
    assert not torch.equal(baseline, perturbed), (
        "output unchanged when the prefix was corrupted -> test is vacuous"
    )
