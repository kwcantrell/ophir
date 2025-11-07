from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    _LARGE_SPARSE_BLOCK_SIZE,
    BlockMask,
    create_block_mask,
    flex_attention,
)

compiled_flex_attention = torch.compile(flex_attention, dynamic=True)

identity_block_mask = BlockMask.from_kv_blocks(
    kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device="cuda"),
    kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device="cuda"),
    BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
    seq_lengths=(1, 1),
)


def generate_alibi_bias():
    alibi_bias = []
    for h in range(4):
        alibi_bias.append(-((h + 1) * 8.0 / 4))
    alibi_bias = torch.tensor(alibi_bias, device="cuda")
    alibi_bias = torch.exp2(alibi_bias)
    return alibi_bias


alibi_bias = generate_alibi_bias()


def alibi_mod(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (kv_idx - q_idx)
    return score + bias


causal_masks = {}


def causal_mod(elm_enc):
    def _inner(b, h, q_idx, kv_idx):
        return ((q_idx <= elm_enc) & (kv_idx <= elm_enc)) | (q_idx >= kv_idx)

    return _inner


def causal_block_mask(elements_in_encoder, NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN):
    key = (elements_in_encoder, NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN)
    if key in causal_masks:
        return causal_masks[key]

    mask = create_block_mask(
        causal_mod(elements_in_encoder), NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN
    )
    causal_masks[key] = mask
    return mask


class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class FlexMHA(nn.Module):
    def __init__(self, num_heads, emb_dim):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # q, k, v projections and output
        self.proj_q = nn.Linear(emb_dim, emb_dim)
        self.proj_k = nn.Linear(emb_dim, emb_dim)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.proj_out = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        B, L, emb_q = query.shape
        _, S, emb_v = value.shape

        # k must have same embedding size as q
        assert key.shape[-1] == emb_q

        # k must have same sequence length as v
        assert key.shape[1] == S

        # current implementation requires q and v to have same embedding size. This requirement may be removed later.
        assert emb_q == emb_v

        # since we currently require q and v to have same emb size, these are the same
        Hq = self.num_heads
        E = self.head_dim
        Hkv = self.num_heads
        Ev = self.head_dim

        # expand dimensions for multihead attention
        query = self.proj_q(query).reshape(B, L, Hq, E).transpose(1, 2)  # output <B, Hq, L, E>
        key = self.proj_k(key).reshape(B, S, Hkv, E).transpose(1, 2)  # output <B, Hkv, S, E>
        value = self.proj_v(value).reshape(B, S, Hkv, Ev).transpose(1, 2)  # output <B, Hkv, S, Ev>

        # attention = compiled_flex_attention(
        #     query, key, value, score_mod=alibi_mod, block_mask=block_mask
        # )
        attention = compiled_flex_attention(query, key, value, block_mask=block_mask)
        # combine heads
        return self.proj_out(attention.transpose(1, 2).reshape(B, L, Hkv * Ev))


class TransformerEncoderBlock(nn.Module):
    __constants__ = ["emb_dim", "num_heads"]
    emb_dim: float
    num_heads: float

    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mha = FlexMHA(num_heads=num_heads, emb_dim=self.emb_dim)
        self.mlp = MLP(self.emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self._rezero)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder = x + self._rezero * self.mha(x, x, x, None)
        return encoder + self._rezero * self.mlp(encoder)


class TransformerDecoderBlock(nn.Module):
    __constants__ = ["emb_dim", "num_heads", "elements_in_encoder"]
    emb_dim: float
    num_heads: float
    elements_in_encoder: int

    def __init__(self, emb_dim, num_heads, elements_in_encoder):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.elements_in_encoder = elements_in_encoder

        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.causal_mha = FlexMHA(num_heads=num_heads, emb_dim=self.emb_dim)
        self.mlp = MLP(self.emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self._rezero)

    def forward(self, decoder: torch.Tensor) -> torch.Tensor:
        causal_mask = causal_block_mask(
            self.elements_in_encoder, None, None, decoder.shape[1], decoder.shape[1]
        )
        decoder = decoder + self._rezero * self.causal_mha(decoder, decoder, decoder, causal_mask)
        return decoder + self._rezero * self.mlp(decoder)


class Transformer(nn.Module):
    __constants__ = [
        "emb_dim",
        "num_end_layers",
        "num_dec_layers",
        "num_heads",
        "encoder",
        "decoder",
    ]
    emb_dim: int
    num_enc_layers: int
    num_dec_layers: int
    num_heads: int

    def __init__(self, emb_dim: int, num_enc_layers: int, num_dec_layers: int, num_heads: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads

        self.encoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.emb_dim, self.num_heads)
                for _ in range(self.num_dec_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.emb_dim, self.num_heads)
                for _ in range(self.num_dec_layers)
            ]
        )

    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            decoder_input = layer(decoder_input, encoder_input)

        for layer in self.decoder:
            decoder_input = layer(decoder_input, encoder_input)
        return decoder_input


class xVal(nn.Module):
    def __init__(self, k, emb_dim, mean: float, std: float):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.weight = nn.Parameter(torch.empty((1, 1, emb_dim)))
        self.ff_out = nn.Linear(emb_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + self.mean) / self.std
        encoding = F.tanh(x) * self.weight
        return encoding

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_out(x)
        x = x * self.std + self.mean
        return x


class RotaryNumberEmbedding(nn.Module):
    def __init__(self, emb_dim: int, min_value: float, max_value: float):
        super().__init__()

        r = max_value - min_value
        self.emb_dim = emb_dim
        exp = (-2 * (emb_dim / 2 - 1)) / emb_dim
        self.base = (torch.pi / r) ** (1 / exp)
        self.wk = nn.Parameter(
            torch.tensor([self.base**exp for _ in range(emb_dim // 2)], dtype=torch.float),
            requires_grad=False,
        )

        self.weight = nn.Parameter(torch.empty((1, 1, emb_dim)))
        self.reset_parameters()

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.set_(weight)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor):
        # create rotation matrices
        x = x.squeeze(-1)
        freqs = torch.einsum("...i,...j->...ij", x, self.wk)
        # Expand to full dimension (cos/sin pairs)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

        even, odd = self.weight[..., ::2], self.weight[..., 1::2]
        weight_rotated = torch.stack([-odd, even], dim=-1).reshape_as(self.weight)
        return self.weight * cos + weight_rotated * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.base = 10000
        self.wk = nn.Parameter(
            1.0 / (self.base ** (torch.arange(0, self.emb_dim, 2).float() / self.emb_dim)),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        # create rotation matrices
        _, s, _ = x.shape
        positions = torch.arange(s, dtype=x.dtype, device=x.device)
        freqs = torch.einsum("i,j->ij", positions, self.wk)

        # Expand to full dimension (cos/sin pairs)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

        even, odd = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([-odd, even], dim=-1).reshape_as(x)
        return x * cos + x_rotated * sin


class SingleCoinPricePredictor(nn.Module):
    __constants__ = [
        "emb_dim",
        "num_dec_layers",
        "num_heads",
        "min_value",
        "max_value",
        "elements_in_encoder",
    ]
    emb_dim: int
    num_dec_layers: int
    num_heads: int
    min_value: float
    max_value: float
    elements_in_encoder: int

    def __init__(
        self,
        emb_dim: int,
        num_dec_layers: int,
        num_heads: int,
        min_value: float,
        max_value: float,
        elements_in_encoder: int,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.min_value = min_value
        self.max_value = max_value
        self.elements_in_encoder = elements_in_encoder

        self.discretizer = RotaryNumberEmbedding(emb_dim, min_value, max_value)
        self.start_emb = nn.Parameter(torch.empty(emb_dim))
        nn.init.normal_(self.start_emb)
        self.mask_emb = nn.Parameter(torch.empty(emb_dim))
        nn.init.normal_(self.mask_emb)

        self.rot_embeddings = RotaryEmbedding(emb_dim)
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.emb_dim, self.num_heads, self.elements_in_encoder)
                for _ in range(self.num_dec_layers)
            ]
        )
        self.ff_out = nn.Linear(emb_dim, 1)

    def _prepare_encoder_input(self, encoder_input: torch.Tensor, mask=None) -> torch.Tensor:
        b, _, _ = encoder_input.shape
        start_token = torch.broadcast_to(self.start_emb, (b, 1, self.emb_dim))

        if self.training and mask is not None:
            b, _, _ = encoder_input.shape
            unmask = torch.rand_like(encoder_input) < 0.2
            change_mask = torch.rand_like(encoder_input) < 0.1

            change = torch.empty_like(encoder_input)
            change.normal_(encoder_input.mean(), encoder_input.std())
            encoder_input = torch.where(mask & unmask & change_mask, change, encoder_input)

            encoder_input = self.discretizer(encoder_input)
            encoder_input = torch.where(mask & ~unmask, self.mask_emb, encoder_input)
        else:
            encoder_input = self.discretizer(encoder_input)

        encoder_input = torch.concat([encoder_input, start_token], dim=1)
        return encoder_input

    def _prepare_decoder_input(self, decoder_input: torch.Tensor, encoder_input) -> torch.Tensor:
        if decoder_input is not None:
            if self.training:
                print("???")
                mask = torch.rand_like(decoder_input) < 0.1
                change = torch.empty_like(decoder_input)
                change.normal_(decoder_input.mean(), decoder_input.std())
                decoder_input = torch.where(mask, change, decoder_input)
            decoder_input = self.discretizer(decoder_input)
            output = torch.cat([encoder_input, decoder_input], dim=1)
        else:
            output = encoder_input
        return output

    def forward(
        self,
        enc_inp: torch.Tensor,
        dec_inp: torch.Tensor,
        random_mask: Optional[torch.Tensor] = None,
    ):
        encoder_input = self._prepare_encoder_input(enc_inp, random_mask)
        decoder_input = self._prepare_decoder_input(dec_inp, encoder_input)

        decoder_output = self.rot_embeddings(decoder_input)

        for layer in self.decoder:
            decoder_output = layer(decoder_output)
        model_output = self.ff_out(decoder_output)

        return model_output
