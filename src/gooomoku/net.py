from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def _pick_attention_heads(channels: int, max_heads: int) -> int:
    capped = max(1, int(max_heads))
    for heads in (8, 6, 4, 3, 2, 1):
        if heads > capped:
            continue
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


def _intermediate_tap_indices(num_blocks: int, stride: int) -> tuple[int, ...]:
    if num_blocks <= 1 or stride <= 0:
        return ()
    return tuple(i for i in range(stride - 1, num_blocks - 1, stride))


class TransformerBlock(nn.Module):
    board_size: int
    channels: int
    num_heads: int
    mlp_hidden: int
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.bfloat16

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
        attn_logits = attn_logits * jnp.asarray(head_dim ** -0.5, dtype=self.compute_dtype)
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

        attn_weights = jax.nn.softmax(attn_logits, axis=-1).astype(self.compute_dtype)
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
    total_blocks: int
    stochastic_depth_rate: float = 0.0
    train: bool = False
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, carry: jnp.ndarray, layer_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        block_out = TransformerBlock(
            board_size=self.board_size,
            channels=self.channels,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(carry)
        residual = block_out - carry
        if self.train and self.stochastic_depth_rate > 0.0:
            layer_ratio = (layer_idx.astype(jnp.float32) + jnp.float32(1.0)) / jnp.float32(max(1, self.total_blocks))
            drop_rate = jnp.float32(self.stochastic_depth_rate) * layer_ratio
            keep_prob = jnp.clip(jnp.float32(1.0) - drop_rate, jnp.float32(1e-3), jnp.float32(1.0))
            sd_key = jax.random.fold_in(self.make_rng("stochastic_depth"), layer_idx.astype(jnp.uint32))
            keep_mask = jax.random.bernoulli(
                sd_key,
                p=keep_prob,
                shape=(carry.shape[0], 1, 1),
            )
            residual = residual * keep_mask.astype(residual.dtype) / keep_prob.astype(residual.dtype)
        out = carry + residual
        return out, out


class PolicyValueNet(nn.Module):
    board_size: int
    channels: int = 64
    blocks: int = 6
    max_attention_heads: int = 4
    stochastic_depth_rate: float = 0.0
    intermediate_supervision_stride: int = 0
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        *,
        return_aux: bool = False,
        return_intermediate: bool = False,
        train: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        layer_outputs = None
        if self.blocks > 0:
            heads = _pick_attention_heads(self.channels, self.max_attention_heads)
            transformer_stack = nn.scan(
                TransformerScanCell,
                variable_axes={"params": 0},
                split_rngs={"params": True, "stochastic_depth": False},
                in_axes=0,
                out_axes=0,
            )
            x, layer_outputs = transformer_stack(
                board_size=self.board_size,
                channels=self.channels,
                num_heads=heads,
                mlp_hidden=self.channels * 4,
                total_blocks=self.blocks,
                stochastic_depth_rate=self.stochastic_depth_rate,
                train=train,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name="transformer_stack",
            )(
                x,
                jnp.arange(self.blocks, dtype=jnp.int32),
            )
        else:
            layer_outputs = jnp.zeros((0, x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)

        x = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="final_norm",
        )(x)

        policy_head = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="policy_head",
        )
        threat_head = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="threat_head",
        )
        legality_head = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="legality_head",
        )
        win1_head = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="win1_head",
        )
        global_pool_norm = nn.LayerNorm(
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_pool_norm",
        )
        global_attn_logits = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_attn_logits",
        )
        global_fuse = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="global_fuse",
        )
        value_dense_0 = nn.Dense(
            self.channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="value_dense_0",
        )
        value_dense_1 = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="value_dense_1",
        )
        horizon_head = nn.Dense(
            1,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="horizon_head",
        )

        def _value_heads(tokens_x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            pooled = global_pool_norm(tokens_x)
            attn_logits = global_attn_logits(pooled).squeeze(-1)
            attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=1).astype(self.compute_dtype)
            attn_pool = jnp.sum(pooled * attn_weights[..., None], axis=1)
            mean_pool = jnp.mean(pooled, axis=1)
            max_pool = jnp.max(pooled, axis=1)
            value_concat = jnp.concatenate([attn_pool, mean_pool, max_pool], axis=-1)
            value_features = global_fuse(value_concat)
            value_features = nn.gelu(value_features, approximate=True)
            value_features = value_dense_0(value_features)
            value_features = nn.gelu(value_features, approximate=True)
            value = value_dense_1(value_features)
            horizon_logit = horizon_head(value_features)
            return value.squeeze(-1), horizon_logit.squeeze(-1)

        policy = policy_head(x).squeeze(-1)
        threat_logits = threat_head(x).squeeze(-1)
        legality_logits = legality_head(x).squeeze(-1)
        win1_logits = win1_head(x).squeeze(-1)
        value, horizon_logit = _value_heads(x)
        value = jnp.tanh(value)
        tap_policy_logits = jnp.zeros((0, x.shape[0], tokens), dtype=self.compute_dtype)
        tap_values = jnp.zeros((0, x.shape[0]), dtype=self.compute_dtype)
        if return_intermediate:
            tap_indices = _intermediate_tap_indices(self.blocks, self.intermediate_supervision_stride)
            if tap_indices:
                tap_x = jnp.take(layer_outputs, jnp.asarray(tap_indices, dtype=jnp.int32), axis=0)
                tap_policy_logits = jax.vmap(lambda tx: policy_head(tx).squeeze(-1))(tap_x)
                tap_value_raw, _ = jax.vmap(_value_heads)(tap_x)
                tap_values = jnp.tanh(tap_value_raw)
        if return_aux:
            if return_intermediate:
                return policy, value, threat_logits, horizon_logit, legality_logits, win1_logits, tap_policy_logits, tap_values
            return policy, value, threat_logits, horizon_logit, legality_logits, win1_logits
        return policy, value
