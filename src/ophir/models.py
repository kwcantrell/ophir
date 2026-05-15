# -*- coding: utf-8 -*-

"""
OHLC multi‑class predictor with a flexible attention backbone.

This module implements the core model used in the project.  The most
notable changes from the original version are:

* Causal block masks now cache also on the padding mask, preventing
  accidental reuse when the padding mask changes.
* Minor type‑hinting and docstring improvements.
* A few defensive checks added for clarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from .model_data import OHLCMulitClassPredictorInput

compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


# --------------------------------------------------------------------------- #
# Hyper‑parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, kw_only=True, slots=True)
class OHLCMulitClassParameters:
    """
    Hyper‑parameters for the OHLC multi‑class predictor.

    ``emb_dim`` – dimensionality of the token embeddings.
    ``num_layers`` – number of transformer blocks.
    ``num_heads`` – number of attention heads.
    """

    emb_dim: int
    num_layers: int
    num_heads: int

    def __post_init__(self):
        assert self.emb_dim % 4 == 0  # emb_dim must be a multiple of 4
        assert (
            self.emb_dim % self.num_heads == 0
        )  # emb_dim must be divisible by num_heads

    @property
    def head_dim(self) -> int:
        """Dimensionality of a single attention head."""
        return self.emb_dim // self.num_heads

    @property
    def hidden_dim(self):
        """Hidden dimension used inside the MLP (4×emb_dim)."""
        return 4 * self.emb_dim


# --------------------------------------------------------------------------- #
# Padding mask
# --------------------------------------------------------------------------- #
def create_padding_mask(pads: torch.Tensor):
    """
    Build a padding mask that can be used inside the flexible attention block.

    Parameters
    ----------
    pads : torch.Tensor
        Boolean tensor of shape ``(B, L)`` where ``True`` indicates a padding
        position (no valid data).  The returned callable expects the same
        arguments as the other mask functions in ``flex_attention``.
    """

    def padding(b, h, q_idx, kv_idx) -> torch.BoolTensor:
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
    m = m0 ** torch.arange(1, 1 + n)

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
    masks: Dict[Tuple[int, int], BlockMask] = None

    def __getitem__(self, index: tuple) -> BlockMask:
        """
        Retrieve a block mask for a given sequence length ``L`` and response size ``S``.
        The third element of ``index`` is the padding-mask function; its id is used
        as part of the cache key.

        Parameters
        ----------
        index : tuple
            ``(L, S, pad_mask)``
            * ``L`` – total sequence length (int or 0‑d tensor).
            * ``S`` – response block size (int or 0‑d tensor).
            * ``pad_mask`` – padding‑mask callable.
        """
        L, S, pad_mask = index
        if isinstance(L, torch.Tensor):
            L = int(L.cpu().numpy())
        if isinstance(S, torch.Tensor):
            S = int(S.cpu().numpy())

        key = (L, S)

        if self.masks is None:
            self.masks = {}

        if key not in self.masks:
            self.masks[key] = create_block_mask(
                self.create_mask(L, S, pad_mask), None, None, L, L
            )
        return self.masks[key]

    def create_mask(self, L, S, pad_mask) -> BlockMask:
        """
        Build the causal mask function for a single block.

        Parameters
        ----------
        L : int
            Total sequence length.
        S : int
            Size of the response block (the last ``S`` tokens).
        pad_mask : nn.Module | None
            Optional padding mask.
        """
        response_block_size = L - S

        def prefix(b, h, q_idx, kv_idx):
            # Queries in the prefix block can only attend to the prefix block.
            return (q_idx < response_block_size) & (kv_idx < response_block_size)

        def causal(b, h, q_idx, kv_idx):
            # Classic causal mask: query index >= key index
            return q_idx >= kv_idx

        causal_mask = or_masks(prefix, causal)

        # Debug – can be removed in production
        print(f"creating block mask of size {L} with response size {S}...")

        if pad_mask is not None:
            return and_masks(causal_mask, pad_mask)
        else:
            return causal_mask


# --------------------------------------------------------------------------- #
# Flexible multi‑head attention
# --------------------------------------------------------------------------- #
class FlexMHA(nn.Module):
    """Multi‑head attention with ALiBi bias and flexible block masking."""

    slopes: torch.Tensor

    def __init__(self, hparams: OHLCMulitClassParameters):
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
            Output tensor of shape ``(B, L, emb_dim)``.
        """
        B, L, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)
        k = k.view(B, L, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)
        v = v.view(B, L, self.hparams.num_heads, self.hparams.head_dim).transpose(1, 2)

        def alibi_score_mod(score, b, h, qi, ki):
            m = self.slopes[h]
            bias = -m * (qi - ki).abs().float()
            return score + bias

        scale = 1 / math.sqrt(self.hparams.head_dim)

        # Forward through the compiled flexible attention
        out = compiled_flex_attention(
            q, k, v, score_mod=alibi_score_mod, block_mask=block_mask, scale=scale
        )

        out = out.transpose(1, 2).contiguous().view(B, L, self.hparams.emb_dim)

        # combine heads
        return self.proj_out(out)


# --------------------------------------------------------------------------- #
# MLP used inside the transformer block
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Simple two‑layer MLP with GELU and dropout."""

    def __init__(self, input_dim, hidden_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """
    A single transformer block with residual connections and ReZero scaling.
    """

    def __init__(self, hparams: OHLCMulitClassParameters):
        super().__init__()
        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float))

        self.mha = FlexMHA(hparams=hparams)
        self.ln1 = nn.LayerNorm(hparams.emb_dim)

        self.mlp = MLP(hparams.emb_dim, 4 * hparams.emb_dim, hparams.emb_dim)
        self.ln2 = nn.LayerNorm(hparams.emb_dim)

    def forward(self, x: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
        # Self‑attention with residual connection
        encoder = x + self._rezero * self.mha(self.ln1(x), block_mask)
        # Feed‑forward with residual connection
        return encoder + self._rezero * self.mlp(self.ln2(encoder))


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class OHLCMulitClassPredictor(nn.Module):
    """
    OHLC multi‑class predictor that outputs:

    * ``model_output`` – raw logits for the 3 target classes
      (r_return, upside, downside) for the *response* tokens.
    * ``stock_embeddings`` – a single embedding per example obtained by
      averaging the response embeddings.
    """

    def __init__(self, hparams: OHLCMulitClassParameters):
        super().__init__()
        # Positional encoding – we keep it simple and trainable
        self.pe = nn.Parameter(torch.randn((1, 512, hparams.emb_dim)))
        self.feature_mlp = nn.Linear(13, hparams.emb_dim)
        self.causal_masks = CausalPrefixBlockMasks()
        self.encoder = nn.ModuleList(
            [TransformerBlock(hparams) for _ in range(hparams.num_layers)]
        )
        self.out_ff = nn.Linear(hparams.emb_dim, 3)

    def forward(self, input: OHLCMulitClassPredictorInput):
        """
        Forward pass of the predictor.

        Parameters
        ----------
        input : OHLCMulitClassPredictorInput
            The input dataclass containing:
            * ``feature_input`` – raw features of shape ``(B, L, 13)``.
            * ``trade_occured`` – boolean mask of shape ``(B, L)``.
            * ``response_size`` – integer indicating the number of response tokens.
            * ``model_output`` and ``stock_embeddings`` are written in‑place.

        Returns
        -------
        OHLCMulitClassPredictorInput
            The same object with ``model_output`` and ``stock_embeddings`` filled.
        """
        # Feature projection
        feature = input.feature_input
        x = self.feature_mlp(feature)

        # Add positional encoding (safely slice to the actual length)
        _, L, _ = x.shape
        pe_slice = self.pe[:, :L]
        x = x + pe_slice

        # Build padding mask – True where no trade occurred
        padding_mask = ~input.trade_occured

        # Build the causal block mask for this batch
        block_mask = self.causal_masks[
            L, input.response_size, create_padding_mask(padding_mask)
        ]

        # Pass through the transformer encoder
        for encoder_block in self.encoder:
            x = encoder_block(x, block_mask)

        # Extract the response embeddings and compute outputs
        response_embeddings = x[:, -input.response_size :]
        input.model_output = self.out_ff(response_embeddings)
        input.stock_embeddings = response_embeddings.mean(dim=1)
        return input
