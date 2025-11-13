from typing import Optional, Tuple

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
        # return ((q_idx <= elm_enc) & (kv_idx <= elm_enc)) | (q_idx >= kv_idx)
        return (q_idx < elm_enc) | (q_idx >= kv_idx)
        # return q_idx >= kv_idx

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

        self.key_cache = None
        self.value_cache = None
        self.cache_index = 0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
        kv_cache: bool = False,
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
        decoder = decoder + self._rezero * self.causal_mha(decoder, decoder, decoder, None)
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
            torch.tensor([self.base**exp for _ in range(emb_dim // 2)]), requires_grad=False
        )
        self.weight = nn.Parameter(torch.empty(1, 1, emb_dim))
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # create rotation matrices
        freqs = torch.einsum("...i,...j->...ij", x.squeeze(-1), self.wk)
        # Expand to full dimension (cos/sin pairs)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

        b, s, _ = x.shape
        weight = self.weight.expand((b, s, -1))
        even, odd = weight[..., ::2], weight[..., 1::2]
        weight_rotated = torch.stack([-odd, even], dim=-1).reshape_as(weight)
        return weight * cos + weight_rotated * sin


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
        self.rot_embeddings = RotaryEmbedding(emb_dim)
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(self.emb_dim, self.num_heads, self.elements_in_encoder)
                for _ in range(self.num_dec_layers)
            ]
        )


class MulitClassPricePredictor(SingleCoinPricePredictor):
    __constants__ = ["num_sector", "num_stock"]
    num_sector: int
    num_stock: int

    def __init__(
        self, num_sector: int, num_industry: int, num_stock: int, **kwargs: SingleCoinPricePredictor
    ):
        kwargs["elements_in_encoder"] = 3
        super().__init__(**kwargs)

        self.num_sector = num_sector
        self.num_industry = num_industry
        self.num_stock = num_stock

        self.year_embedding = nn.Embedding(100, self.emb_dim)
        self.month_embedding = nn.Embedding(15, self.emb_dim)
        self.day_embedding = nn.Embedding(33, self.emb_dim)
        self.sector_embedding = nn.Embedding(num_sector, self.emb_dim)
        self.industry_embedding = nn.Embedding(num_industry, self.emb_dim)
        self.stock_embedding = nn.Embedding(num_stock, self.emb_dim)
        self.stock_dec_embedding = nn.Embedding(num_stock, self.emb_dim)
        self.start_embedding = nn.Embedding(1, self.emb_dim)
        self.mask_embedding = nn.Embedding(1, self.emb_dim)

        self.ff_out = nn.Linear(self.emb_dim, 1)
        self.year_ff = nn.Linear(self.emb_dim, 100)
        self.month_ff = nn.Linear(self.emb_dim, 15)
        self.day_ff = nn.Linear(self.emb_dim, 33)
        self.sector_ff = nn.Linear(self.emb_dim, num_sector)
        self.industry_ff = nn.Linear(self.emb_dim, num_industry)
        self.stock_ff = nn.Linear(self.emb_dim, num_stock)

    def _change_date(
        self,
        token: torch.Tensor,
        min,
        max,
        embedding_module: nn.Module,
        change_mask: torch.Tensor,
        unmask: torch.Tensor,
        mask: torch.Tensor,
    ):
        return embedding_module(
            torch.where(
                (mask & unmask & change_mask).squeeze(-1),
                torch.randint_like(token, min, max + 1).long(),
                token,
            )
        )

    def _change_ticker_info(self, token: torch.Tensor, embedding_module: nn.Module, mask):
        return embedding_module(
            torch.where(
                mask,
                torch.randint_like(token, token.min(), token.max() + 1).long(),
                token,
            )
        )

    def _prepare_encoder_input(self, stock: Tuple[torch.Tensor], mask: bool = None) -> torch.Tensor:
        (stock_value, year_token, month_token, day_token, mask_embedding) = stock

        if self.training and mask is not None:
            unmask = torch.rand_like(stock_value) < 0.2
            change_mask = torch.rand_like(stock_value) < 0.1

            random_value = torch.empty_like(stock_value)
            random_value.normal_(stock_value.mean(), stock_value.std())

            encoder_embedding = torch.where(
                mask & ~unmask,
                mask_embedding,
                self.discretizer(
                    torch.where(mask & unmask & change_mask, random_value, stock_value)
                )
                + self._change_date(
                    year_token, 0, 99, self.year_embedding, change_mask, unmask, mask
                )
                + self._change_date(
                    month_token, 0, 11, self.month_embedding, change_mask, unmask, mask
                )
                + self._change_date(
                    day_token, 0, 30, self.day_embedding, change_mask, unmask, mask
                ),
            )
        else:
            encoder_embedding = (
                self.discretizer(stock_value)
                + self.year_embedding(year_token)
                + self.month_embedding(month_token)
                + self.day_embedding(day_token)
            )
        return encoder_embedding

    def _prepare_decoder_input(self, stock_info: Tuple[torch.Tensor]) -> torch.Tensor:
        (stock_emb, year_token, month_token, day_token) = stock_info
        decoder_input = (
            stock_emb
            + self.year_embedding(year_token)
            + self.month_embedding(month_token)
            + self.day_embedding(day_token)
        )
        return decoder_input

    def forward(
        self,
        encoder_value: torch.Tensor,
        decoder_value: torch.Tensor,
        year: torch.Tensor,
        month: torch.Tensor,
        day: torch.Tensor,
        sector_token: torch.Tensor,
        industry_token: torch.Tensor,
        stock_token: torch.Tensor,
        random_mask: Optional[torch.Tensor] = None,
    ):
        _, enc_len, _ = encoder_value.shape
        mask_embedding = self.mask_embedding(torch.zeros_like(year[:, :enc_len]))
        decoder_embedding = self._prepare_decoder_input(
            (
                self.stock_dec_embedding(stock_token),
                year[:, enc_len:],
                month[:, enc_len:],
                day[:, enc_len:],
            )
        )
        if self.training and random_mask is not None:
            embeddings = self.rot_embeddings(
                torch.concat(
                    [
                        self._change_ticker_info(
                            sector_token, self.sector_embedding, random_mask[:, 0]
                        ),
                        self._change_ticker_info(
                            industry_token, self.industry_embedding, random_mask[:, 1]
                        ),
                        self._change_ticker_info(
                            stock_token, self.stock_embedding, random_mask[:, 2]
                        ),
                        self._prepare_encoder_input(
                            (
                                encoder_value,
                                year[:, :enc_len],
                                month[:, :enc_len],
                                day[:, :enc_len],
                                mask_embedding,
                            ),
                            random_mask,
                        ),
                        decoder_embedding,
                    ],
                    dim=1,
                )
            )
        else:
            embeddings = self.rot_embeddings(
                torch.concat(
                    [
                        self.sector_embedding(sector_token),
                        self.industry_embedding(industry_token),
                        self.stock_embedding(stock_token),
                        self._prepare_encoder_input(
                            (
                                encoder_value,
                                year[:, :enc_len],
                                month[:, :enc_len],
                                day[:, :enc_len],
                                None,
                            )
                        ),
                        decoder_embedding,
                    ],
                    dim=1,
                )
            )

        for layer in self.decoder:
            embeddings = layer(embeddings)

        return (
            self.ff_out(embeddings[:, 3:]),
            self.year_ff(embeddings[:, 3:]),
            self.month_ff(embeddings[:, 3:]),
            self.day_ff(embeddings[:, 3:]),
            self.sector_ff(embeddings[:, 0]),
            self.industry_ff(embeddings[:, 1]),
            self.stock_ff(embeddings[:, 2]),
        )
