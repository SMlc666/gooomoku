from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x + residual)
        return x


class ResidualScanCell(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, carry: jnp.ndarray, _: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        carry = ResidualBlock(channels=self.channels)(carry)
        return carry, None


class PolicyValueNet(nn.Module):
    board_size: int
    channels: int = 64
    blocks: int = 6

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(obs)
        x = nn.relu(x)

        residual_stack = nn.scan(
            ResidualScanCell,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
            out_axes=0,
        )
        x, _ = residual_stack(channels=self.channels, name="residual_stack")(
            x,
            jnp.arange(self.blocks, dtype=jnp.int32),
        )

        policy = nn.Conv(2, kernel_size=(1, 1), padding="SAME")(x)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))
        policy = nn.Dense(self.board_size * self.board_size)(policy)

        value = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(x)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(self.channels)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = jnp.tanh(value).squeeze(-1)
        return policy, value
