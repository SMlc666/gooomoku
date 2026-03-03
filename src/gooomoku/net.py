from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def _pick_attention_heads(channels: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if channels % heads == 0:
            return heads
    return 1


def _relative_position_index(board_size: int) -> jnp.ndarray:
    coords = jnp.arange(board_size, dtype=jnp.int32)
    row, col = jnp.meshgrid(coords, coords, indexing="ij")
    flat = jnp.stack((row.reshape(-1), col.reshape(-1)), axis=1)
    rel = flat[:, None, :] - flat[None, :, :]
    span = 2 * board_size - 1
    delta_row = rel[..., 0] + (board_size - 1)
    delta_col = rel[..., 1] + (board_size - 1)
    return delta_row * span + delta_col


class TransformerBlock(nn.Module):
    board_size: int
    channels: int
    num_heads: int
    mlp_hidden: int
    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"channels={self.channels} must be divisible by num_heads={self.num_heads}"
            )
        head_dim = self.channels // self.num_heads
        y = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)

        q = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="q_proj",
        )(y)
        k = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="k_proj",
        )(y)
        v = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="v_proj",
        )(y)

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k)
        attn_logits = attn_logits * jnp.float32(head_dim ** -0.5)
        rel_table_size = (2 * self.board_size - 1) * (2 * self.board_size - 1)
        rel_pos_bias = self.param(
            "rel_pos_bias",
            nn.initializers.normal(stddev=0.02),
            (self.num_heads, rel_table_size),
            self.param_dtype,
        )
        rel_index = _relative_position_index(self.board_size)
        tokens = self.board_size * self.board_size
        rel_bias = jnp.take(rel_pos_bias.astype(self.compute_dtype), rel_index.reshape(-1), axis=1)
        rel_bias = rel_bias.reshape(self.num_heads, tokens, tokens)
        attn_logits = attn_logits + rel_bias[None, :, :, :]

        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(self.compute_dtype)
        context = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
        context = context.reshape((context.shape[0], context.shape[1], self.channels))
        y = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="attn_out",
        )(context)
        x = x + y

        y = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        y = nn.Dense(
            self.mlp_hidden,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(y)
        y = nn.gelu(y, approximate=True)
        y = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(y)
        return x + y


class TransformerScanCell(nn.Module):
    board_size: int
    channels: int
    num_heads: int
    mlp_hidden: int
    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, carry: jnp.ndarray, _: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        carry = TransformerBlock(
            board_size=self.board_size,
            channels=self.channels,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(carry)
        return carry, None


class PolicyValueNet(nn.Module):
    board_size: int
    channels: int = 64
    blocks: int = 6
    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        tokens = self.board_size * self.board_size
        x = obs.astype(self.compute_dtype).reshape((obs.shape[0], tokens, obs.shape[-1]))
        x = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="token_embed",
        )(x)

        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (tokens, self.channels),
            self.param_dtype,
        )
        x = x + pos_embedding[None, :, :].astype(self.compute_dtype)

        if self.blocks > 0:
            heads = _pick_attention_heads(self.channels)
            transformer_stack = nn.scan(
                TransformerScanCell,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=0,
                out_axes=0,
            )
            x, _ = transformer_stack(
                board_size=self.board_size,
                channels=self.channels,
                num_heads=heads,
                mlp_hidden=self.channels * 4,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name="transformer_stack",
            )(
                x,
                jnp.arange(self.blocks, dtype=jnp.int32),
            )

        x = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="final_norm",
        )(x)

        policy = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="policy_head",
        )(x).squeeze(-1)

        pooled = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_pool_norm",
        )(x)
        attn_logits = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_attn_logits",
        )(pooled).squeeze(-1)
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=1).astype(self.compute_dtype)
        attn_pool = jnp.sum(pooled * attn_weights[..., None], axis=1)
        mean_pool = jnp.mean(pooled, axis=1)
        max_pool = jnp.max(pooled, axis=1)
        value = jnp.concatenate([attn_pool, mean_pool, max_pool], axis=-1)
        value = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_fuse",
        )(value)
        value = nn.gelu(value, approximate=True)
        value = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="value_dense_0",
        )(value)
        value = nn.gelu(value, approximate=True)
        value = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="value_dense_1",
        )(value)
        value = jnp.tanh(value).squeeze(-1)
        return policy, value
