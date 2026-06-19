"""
OHLC multi-class predictor with a flexible attention backbone.

This module implements the core model used in the project.  The most
notable changes from the original version are:

* Causal block masks now cache also on the padding mask, preventing
  accidental reuse when the padding mask changes.
* Minor type-hinting and docstring improvements.
* A few defensive checks added for clarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .model_data import OHLCMulitClassPredictorInput

    MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ScoreMod = Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]

compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


# --------------------------------------------------------------------------- #
# Hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, kw_only=True, slots=True)
class OHLCMulitClassParameters:
    """
    Hyper-parameters for the OHLC multi-class predictor.

    ``emb_dim`` - dimensionality of the token embeddings.
    ``num_layers`` - number of transformer blocks.
    ``num_heads`` - number of attention heads.
    """

    emb_dim: int
    num_layers: int
    num_heads: int

    def __post_init__(self) -> None:
        assert self.emb_dim % 4 == 0  # emb_dim must be a multiple of 4
        assert self.emb_dim % self.num_heads == 0  # emb_dim must be divisible by num_heads

    @property
    def head_dim(self) -> int:
        """Dimensionality of a single attention head."""
        return self.emb_dim // self.num_heads

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension used inside the MLP (4xemb_dim)."""
        return 4 * self.emb_dim


# --------------------------------------------------------------------------- #
# Padding mask
# --------------------------------------------------------------------------- #
def create_padding_mask(pads: torch.Tensor) -> MaskMod:
    """
    Build a padding mask that can be used inside the flexible attention block.

    Parameters
    ----------
    pads : torch.Tensor
        Boolean tensor of shape ``(B, L)`` where ``True`` indicates a padding
        position (no valid data).  The returned callable expects the same
        arguments as the other mask functions in ``flex_attention``.
    """

    def padding(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # The mask is True when *both* positions are not padding.
        return ~pads[b, q_idx] & ~pads[b, kv_idx]

    return padding


## --------------------------------------------------------------------------- #
# ALiBi slopes
# --------------------------------------------------------------------------- #
def get_alibi_slopes(hparams: OHLCMulitClassParameters) -> torch.Tensor:
    """
    Compute the ALiBi slopes for the given number of heads.

    The implementation follows the original ALiBi paper and the
    reference implementation from the labml repo.
    """
    # Closest lower power of two
    n = 2 ** math.floor(math.log2(hparams.num_heads))

    m0 = 2.0 ** (-8.0 / n)
    m = cast("torch.Tensor", m0 ** torch.arange(1, 1 + n))

    # If num_heads is not a power of two, append extra slopes (as in the reference code)
    if n < hparams.num_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = m_hat_0 ** torch.arange(1, 1 + (hparams.num_heads - n), 2)
        m = torch.cat([m, m_hat], dim=0)

    return m


# --------------------------------------------------------------------------- #
# Causal block masks
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True, slots=True)
class CausalPrefixBlockMasks:
    """Lazily built, cached causal + prefix flex-attention block masks.

    Masks are keyed by ``(seq_len, response_size)`` so that a given shape is
    only constructed once and reused across forward passes.
    """

    masks: dict[tuple[int, int], BlockMask] | None = None

    def __getitem__(
        self,
        index: tuple[int | torch.Tensor, int | torch.Tensor, MaskMod],
    ) -> BlockMask:
        """
        Retrieve a block mask for a given sequence length ``L`` and response size ``S``.
        The third element of ``index`` is the padding-mask function; its id is used
        as part of the cache key.

        Parameters
        ----------
        index : tuple
            ``(seq_len, response_size, pad_mask)``
            * ``seq_len`` - total sequence length (int or 0-d tensor).
            * ``response_size`` - response block size (int or 0-d tensor).
            * ``pad_mask`` - padding-mask callable.
        """
        seq_len_in, response_size_in, pad_mask = index
        seq_len = (
            int(seq_len_in.cpu().numpy()) if isinstance(seq_len_in, torch.Tensor) else seq_len_in
        )
        response_size = (
            int(response_size_in.cpu().numpy())
            if isinstance(response_size_in, torch.Tensor)
            else response_size_in
        )

        key = (seq_len, response_size)

        if self.masks is None:
            self.masks = {}

        if key not in self.masks:
            self.masks[key] = create_block_mask(
                self.create_mask(seq_len, response_size, pad_mask), None, None, seq_len, seq_len
            )
        return self.masks[key]

    def create_mask(self, seq_len: int, response_size: int, pad_mask: MaskMod | None) -> MaskMod:
        """
        Build the causal mask function for a single block.

        Parameters
        ----------
        seq_len : int
            Total sequence length.
        response_size : int
            Size of the response block (the last ``response_size`` tokens).
        pad_mask : callable | None
            Optional padding mask.
        """
        response_block_size = seq_len - response_size

        def prefix(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            # Queries in the prefix block can only attend to the prefix block.
            return (q_idx < response_block_size) & (kv_idx < response_block_size)

        def causal(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            # Classic causal mask: query index >= key index
            return q_idx >= kv_idx

        causal_mask = or_masks(prefix, causal)

        # Debug - can be removed in production
        print(f"creating block mask of size {seq_len} with response size {response_size}...")

        if pad_mask is not None:
            return and_masks(causal_mask, pad_mask)
        else:
            return causal_mask


# --------------------------------------------------------------------------- #
# Flexible multi-head attention
# --------------------------------------------------------------------------- #
class FlexMHA(nn.Module):
    """Multi-head attention with ALiBi bias and flexible block masking."""

    slopes: torch.Tensor

    def __init__(self, hparams: OHLCMulitClassParameters) -> None:
        super().__init__()
        self.hparams = hparams

        # ALiBi slopes
        self.register_buffer("slopes", get_alibi_slopes(hparams))

        self.qkv = nn.Linear(hparams.emb_dim, 3 * hparams.emb_dim, bias=False)
        self.proj_out = nn.Linear(hparams.emb_dim, hparams.emb_dim)

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        """
        Forward pass of the flexible MHA.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, emb_dim)``.
        block_mask : BlockMask
            Mask object created by :class:`CausalBlockMasks`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, emb_dim)``.
        """
        batch, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)

        def alibi_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            qi: torch.Tensor,
            ki: torch.Tensor,
        ) -> torch.Tensor:
            m = self.slopes[h]
            bias = -m * (qi - ki).abs().float()
            return score + bias

        scale = 1 / math.sqrt(self.hparams.head_dim)

        # Forward through the compiled flexible attention
        out = cast(
            "torch.Tensor",
            compiled_flex_attention(
                q, k, v, score_mod=alibi_score_mod, block_mask=block_mask, scale=scale
            ),
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hparams.emb_dim)

        # combine heads
        return cast("torch.Tensor", self.proj_out(out))


# --------------------------------------------------------------------------- #
# MLP used inside the transformer block
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Simple two-layer MLP with GELU and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, L, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, L, emb_dim)``.
        """
        return cast("torch.Tensor", self.mlp(x))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """
    A single transformer block with residual connections and ReZero scaling.
    """

    def __init__(self, hparams: OHLCMulitClassParameters) -> None:
        super().__init__()
        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float))

        self.mha = FlexMHA(hparams=hparams)
        self.ln1 = nn.LayerNorm(hparams.emb_dim)

        self.mlp = MLP(hparams.emb_dim, 4 * hparams.emb_dim, hparams.emb_dim)
        self.ln2 = nn.LayerNorm(hparams.emb_dim)

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        """Apply attention then MLP, each in a ReZero-scaled residual.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, L, emb_dim)``.
        block_mask : BlockMask
            Mask from :class:`CausalPrefixBlockMasks`.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, L, emb_dim)``.
        """
        # Self-attention with residual connection
        encoder = x + self._rezero * self.mha(self.ln1(x), block_mask)
        # Feed-forward with residual connection
        return cast("torch.Tensor", encoder + self._rezero * self.mlp(self.ln2(encoder)))


# --------------------------------------------------------------------------- #
# Output activations
# --------------------------------------------------------------------------- #
def apply_output_activations(raw: torch.Tensor) -> torch.Tensor:
    """Constrain the upside/downside channels to be non-negative.

    ``upside``/``downside`` are log-magnitudes (``log(high/close)`` and
    ``log(close/low)``), both ``>= 0`` by construction, while ``r_close`` is a
    signed return. Softplus keeps the magnitude channels in their valid range so
    the ``.exp()`` reconstruction can never invert the candle.
    """
    r_close = raw[..., 0:1]
    upside = F.softplus(raw[..., 1:2])
    downside = F.softplus(raw[..., 2:3])
    return torch.cat([r_close, upside, downside], dim=-1)


def pool_prefix_embedding(x: torch.Tensor, response_size: int) -> torch.Tensor:
    """Mean-pool the prefix (observed-history) positions into one vector/example.

    Pools ``x[:, :-response_size]`` — the positions that carry real features —
    rather than the masked forecast block, giving a more grounded per-stock
    embedding for the UI projection.
    """
    return x[:, :-response_size].mean(dim=1)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class OHLCMulitClassPredictor(nn.Module):
    """
    OHLC multi-class predictor that outputs:

    * ``model_output`` - three forward regression targets for the *response*
      tokens: relative close return (r_close), plus softplus-constrained
      non-negative log-magnitudes for upside and downside.
    * ``stock_embeddings`` - a single embedding per example obtained by
      mean-pooling the prefix (observed-history) embeddings.
    """

    def __init__(self, hparams: OHLCMulitClassParameters) -> None:
        super().__init__()
        # Positional encoding - we keep it simple and trainable
        self.pe = nn.Parameter(torch.randn((1, 512, hparams.emb_dim)))
        self.feature_mlp = nn.Linear(12, hparams.emb_dim)
        # Learned token that replaces the response-block features so the model
        # cannot read the targets it is asked to forecast (see _apply_response_mask).
        self.mask_token = nn.Parameter(torch.randn(hparams.emb_dim))
        self.causal_masks = CausalPrefixBlockMasks()
        self.encoder = nn.ModuleList([TransformerBlock(hparams) for _ in range(hparams.num_layers)])
        self.out_ff = nn.Linear(hparams.emb_dim, 3)

    def _apply_response_mask(self, x: torch.Tensor, response_size: int) -> torch.Tensor:
        """Replace the response-block rows of ``x`` with the learned mask token.

        The last ``response_size`` positions are the days the model must
        forecast. Every input feature at those positions is contemporaneous
        with that day's targets (``r_close`` / ``upside`` / ``downside``, plus
        the rolling features derived from them), so feeding them would leak the
        answer. They are overwritten with a single learned mask embedding,
        leaving only positional information for the forecast horizon.

        Parameters
        ----------
        x : torch.Tensor
            Projected features of shape ``(B, L, emb_dim)``.
        response_size : int
            Number of trailing positions to mask.

        Returns
        -------
        torch.Tensor
            ``x`` with its last ``response_size`` rows replaced by the mask
            token.
        """
        batch, seq_len, _ = x.shape
        prefix_len = seq_len - response_size
        mask = self.mask_token.expand(batch, response_size, -1)
        return torch.cat([x[:, :prefix_len], mask], dim=1)

    def forward(self, input: OHLCMulitClassPredictorInput) -> OHLCMulitClassPredictorInput:
        """
        Forward pass of the predictor.

        Parameters
        ----------
        input : OHLCMulitClassPredictorInput
            The input dataclass containing:
            * ``feature_input`` - raw features of shape ``(B, L, 12)``.
            * ``trade_occured`` - boolean mask of shape ``(B, L)``.
            * ``response_size`` - integer indicating the number of response tokens.
            * ``model_output`` and ``stock_embeddings`` are written in-place.

        Returns
        -------
        OHLCMulitClassPredictorInput
            The same object with ``model_output`` and ``stock_embeddings`` filled.
        """
        # Feature projection
        feature = input.feature_input
        x = cast("torch.Tensor", self.feature_mlp(feature))

        # Mask the forecast horizon so the response days carry no features that
        # would leak their own targets.
        _, seq_len, _ = x.shape
        x = self._apply_response_mask(x, int(input.response_size))

        # Add positional encoding (safely slice to the actual length)
        pe_slice = self.pe[:, :seq_len]
        x = x + pe_slice

        # Build padding mask - True where no trade occurred
        padding_mask = ~input.trade_occured

        # Build the causal block mask for this batch
        block_mask = self.causal_masks[
            seq_len, input.response_size, create_padding_mask(padding_mask)
        ]

        # Pass through the transformer encoder
        for encoder_block in self.encoder:
            x = encoder_block(x, block_mask)

        # Extract the response embeddings and compute outputs
        response_embeddings = x[:, -input.response_size :]
        input.model_output = apply_output_activations(
            cast("torch.Tensor", self.out_ff(response_embeddings))
        )
        input.stock_embeddings = pool_prefix_embedding(x, int(input.response_size))
        return input
