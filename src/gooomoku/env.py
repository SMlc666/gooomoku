from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class GomokuState:
    board: jnp.ndarray
    to_play: jnp.ndarray
    last_action: jnp.ndarray
    num_moves: jnp.ndarray
    terminated: jnp.ndarray
    winner: jnp.ndarray


def reset(board_size: int) -> GomokuState:
    return GomokuState(
        board=jnp.zeros((board_size, board_size), dtype=jnp.int8),
        to_play=jnp.int8(1),
        last_action=jnp.int32(-1),
        num_moves=jnp.int32(0),
        terminated=jnp.bool_(False),
        winner=jnp.int8(0),
    )


def legal_action_mask(state: GomokuState) -> jnp.ndarray:
    legal = (state.board.reshape(-1) == 0).astype(jnp.bool_)
    return jnp.where(state.terminated, jnp.zeros_like(legal), legal)


def _stone_at(board: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
    board_size = board.shape[0]
    in_bounds = (row >= 0) & (row < board_size) & (col >= 0) & (col < board_size)
    return jax.lax.cond(
        in_bounds,
        lambda rc: board[rc[0], rc[1]],
        lambda _: jnp.int8(0),
        operand=(row, col),
    )


def _count_in_direction(
    board: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    dr: int,
    dc: int,
    player: jnp.ndarray,
) -> jnp.ndarray:
    dr32 = jnp.int32(dr)
    dc32 = jnp.int32(dc)

    def cond_fn(carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        r, c, _ = carry
        return _stone_at(board, r, c) == player

    def body_fn(carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r, c, count = carry
        return (r + dr32, c + dc32, count + jnp.int32(1))

    start = (row + dr32, col + dc32, jnp.int32(0))
    _, _, total = jax.lax.while_loop(cond_fn, body_fn, start)
    return total


def _has_five(
    board: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    player: jnp.ndarray,
) -> jnp.ndarray:
    win = jnp.bool_(False)
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        streak = (
            jnp.int32(1)
            + _count_in_direction(board, row, col, dr, dc, player)
            + _count_in_direction(board, row, col, -dr, -dc, player)
        )
        win = win | (streak >= 5)
    return win


def step(state: GomokuState, action: jnp.ndarray) -> Tuple[GomokuState, jnp.ndarray, jnp.ndarray]:
    board_size = state.board.shape[0]
    num_actions = board_size * board_size
    action = jnp.int32(action)

    legal = legal_action_mask(state)
    in_range = (action >= 0) & (action < num_actions)
    legal_at_action = jax.lax.cond(
        in_range,
        lambda a: legal[a],
        lambda _: jnp.bool_(False),
        operand=action,
    )

    can_play = (~state.terminated) & legal_at_action
    illegal_move = (~state.terminated) & (~legal_at_action)

    row = action // board_size
    col = action % board_size

    board_after = jax.lax.cond(
        can_play,
        lambda b: b.at[row, col].set(state.to_play),
        lambda b: b,
        operand=state.board,
    )

    win = jax.lax.cond(
        can_play,
        lambda _: _has_five(board_after, row, col, state.to_play),
        lambda _: jnp.bool_(False),
        operand=None,
    )

    num_moves = state.num_moves + can_play.astype(jnp.int32)
    draw = (~win) & (num_moves >= num_actions)
    terminated = state.terminated | win | draw | illegal_move

    winner = jnp.where(
        win,
        state.to_play,
        jnp.where(illegal_move, -state.to_play, state.winner),
    ).astype(jnp.int8)
    reward = jnp.where(win, jnp.float32(1.0), jnp.where(illegal_move, jnp.float32(-1.0), jnp.float32(0.0)))

    next_to_play = jnp.where(can_play & (~terminated), -state.to_play, state.to_play).astype(jnp.int8)
    last_action = jnp.where(can_play, action, state.last_action).astype(jnp.int32)

    next_state = GomokuState(
        board=board_after,
        to_play=next_to_play,
        last_action=last_action,
        num_moves=num_moves.astype(jnp.int32),
        terminated=terminated.astype(jnp.bool_),
        winner=winner,
    )
    return next_state, reward, next_state.terminated


def encode_state(state: GomokuState) -> jnp.ndarray:
    board = state.board
    board_size = board.shape[0]
    num_actions = board_size * board_size

    mine = (board == state.to_play).astype(jnp.float32)
    opp = (board == -state.to_play).astype(jnp.float32)

    safe_last_action = jnp.clip(state.last_action, 0, num_actions - 1)
    last_plane = jax.nn.one_hot(safe_last_action, num_actions, dtype=jnp.float32).reshape(board_size, board_size)
    last_plane = jnp.where(state.last_action >= 0, last_plane, jnp.zeros_like(last_plane))

    side_to_move = jnp.full((board_size, board_size), (state.to_play == 1).astype(jnp.float32), dtype=jnp.float32)
    return jnp.stack([mine, opp, last_plane, side_to_move], axis=-1)


batch_legal_action_mask = jax.vmap(legal_action_mask)
batch_encode_states = jax.vmap(encode_state)
batch_step = jax.vmap(step)
