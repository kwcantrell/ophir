from typing import Tuple

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


def causal_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def causal_block_mask(NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN):
    key = (NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN)
    if key in causal_masks:
        return causal_masks[key]

    mask = create_block_mask(causal_mod, NUM_BATCH, NUM_HEADS, NUM_Q_LEN, KV_LEN)
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
    __constants__ = ["emb_dim", "num_heads"]
    emb_dim: float
    num_heads: float

    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.causal_mha = FlexMHA(num_heads=num_heads, emb_dim=self.emb_dim)
        self.cross_mha = FlexMHA(num_heads=num_heads, emb_dim=self.emb_dim)
        self.mlp = MLP(self.emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self._rezero)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        causal_mask = causal_block_mask(None, None, decoder.shape[1], decoder.shape[1])
        decoder = decoder + self._rezero * self.causal_mha(decoder, decoder, decoder, causal_mask)

        decoder = decoder + self._rezero * self.cross_mha(decoder, encoder, encoder, None)
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
    def __init__(self, emb_dim, min: float, max: float):
        super().__init__()

        r = max - min
        self.emb_dim = emb_dim
        self.base = r / torch.pi
        self.wk = nn.Parameter(
            1.0 / (self.base ** (torch.arange(0, self.emb_dim, 2).float() / self.emb_dim)),
            requires_grad=False,
        )
        self.weight = nn.Parameter(torch.empty((1, 1, emb_dim)))
        self.reset_parameters()

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
        b, s, e = x.shape
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
        "num_end_layers",
        "num_dec_layers",
        "num_heads",
        "mean",
        "std",
    ]
    emb_dim: int
    num_enc_layers: int
    num_dec_layers: int
    num_heads: int
    mean: float
    std: float

    def __init__(
        self,
        emb_dim: int,
        num_enc_layers: int,
        num_dec_layers: int,
        num_heads: int,
        mean: float,
        std: float,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.mean = mean
        self.std = std

        # self.discretizer = xVal(k=1, emb_dim=self.emb_dim, mean=self.mean, std=self.std)
        size = 500
        num_points = 2 * size + 1
        self.discretizer = RotaryNumberEmbedding(emb_dim, -size, size)
        self.rot_pos = RotaryEmbedding(emb_dim)
        # self.predictor = Transformer(self.emb_dim, self.num_enc_layers, self.num_dec_layers, self.num_heads)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(self.emb_dim, self.num_heads)
                for _ in range(self.num_dec_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.emb_dim, self.num_heads)
                for _ in range(self.num_dec_layers)
            ]
        )
        self.ff_out = nn.Linear(emb_dim, num_points)

    def _prepare_encoder_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.discretizer(input)
        # return input
        return self.rot_pos(input)

    def _prepare_decoder_input(self, input: torch.Tensor, encoder_output) -> torch.Tensor:
        # output = torch.cat([encoder_output, encoder_output[:, -1:, :]], dim=1)
        output = encoder_output[:, -1:, :]
        if input is not None:
            input = self.discretizer(input)
            output = torch.cat([output, input], dim=1)
        return self.rot_pos(output)

    def forward(self, enc_inp: torch.Tensor, dec_inp: torch.Tensor):
        encoder_input = self._prepare_encoder_input(enc_inp)
        encoder_output = encoder_input
        for layer in self.encoder:
            encoder_output = layer(encoder_output)

        decoder_input = self._prepare_decoder_input(dec_inp, encoder_output)
        decoder_output = decoder_input
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output)

        output = self.ff_out(decoder_output)
        return output
