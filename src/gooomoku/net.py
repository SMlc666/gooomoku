from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def _pick_attention_heads(channels: int) -> int:
    for heads in (8, 6, 4, 3, 2, 1):
        if channels % heads == 0:
            return heads
    return 1


class TransformerBlock(nn.Module):
    channels: int
    num_heads: int
    mlp_hidden: int
    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.LayerNorm(
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.channels,
            out_features=self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            dropout_rate=0.0,
        )(y, y, deterministic=True)
        x = x + y

        y = nn.LayerNorm(
            compute_dtype=self.compute_dtype,
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
    channels: int
    num_heads: int
    mlp_hidden: int
    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, carry: jnp.ndarray, _: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        carry = TransformerBlock(
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
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="final_norm",
        )(x)

        policy = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="policy_head",
        )(x).squeeze(-1)

        value = jnp.mean(x, axis=1)
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
