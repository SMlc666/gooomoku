from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class GomokuState:
    board: jnp.ndarray
    black_bits: jnp.ndarray
    white_bits: jnp.ndarray
    to_play: jnp.ndarray
    last_action: jnp.ndarray
    num_moves: jnp.ndarray
    terminated: jnp.ndarray
    winner: jnp.ndarray


@functools.lru_cache(maxsize=None)
def _line_start_masks(board_size: int):
    num_actions = board_size * board_size
    east = np.zeros((num_actions,), dtype=np.bool_)
    south = np.zeros((num_actions,), dtype=np.bool_)
    diag_dr = np.zeros((num_actions,), dtype=np.bool_)
    diag_dl = np.zeros((num_actions,), dtype=np.bool_)

    if board_size >= 5:
        idx = np.arange(num_actions, dtype=np.int32).reshape(board_size, board_size)
        east[idx[:, : board_size - 4].reshape(-1)] = True
        south[idx[: board_size - 4, :].reshape(-1)] = True
        diag_dr[idx[: board_size - 4, : board_size - 4].reshape(-1)] = True
        diag_dl[idx[: board_size - 4, 4:].reshape(-1)] = True

    return (
        jnp.asarray(east),
        jnp.asarray(south),
        jnp.asarray(diag_dr),
        jnp.asarray(diag_dl),
    )


def _shift_bits(bits: jnp.ndarray, offset: int) -> jnp.ndarray:
    if offset <= 0:
        return bits
    zeros = jnp.zeros((offset,), dtype=jnp.bool_)
    return jnp.concatenate([bits[offset:], zeros], axis=0)


def _has_line(bits: jnp.ndarray, starts: jnp.ndarray, offset: int) -> jnp.ndarray:
    return jnp.any(
        starts
        & bits
        & _shift_bits(bits, offset)
        & _shift_bits(bits, 2 * offset)
        & _shift_bits(bits, 3 * offset)
        & _shift_bits(bits, 4 * offset)
    )


def _has_five_from_bits(bits: jnp.ndarray, board_size: int) -> jnp.ndarray:
    east_starts, south_starts, dr_starts, dl_starts = _line_start_masks(board_size)
    east = _has_line(bits, east_starts, 1)
    south = _has_line(bits, south_starts, board_size)
    diag_dr = _has_line(bits, dr_starts, board_size + 1)
    diag_dl = _has_line(bits, dl_starts, board_size - 1)
    return east | south | diag_dr | diag_dl


def reset(board_size: int) -> GomokuState:
    num_actions = board_size * board_size
    return GomokuState(
        board=jnp.zeros((board_size, board_size), dtype=jnp.int8),
        black_bits=jnp.zeros((num_actions,), dtype=jnp.bool_),
        white_bits=jnp.zeros((num_actions,), dtype=jnp.bool_),
        to_play=jnp.int8(1),
        last_action=jnp.int32(-1),
        num_moves=jnp.int32(0),
        terminated=jnp.bool_(False),
        winner=jnp.int8(0),
    )


def legal_action_mask(state: GomokuState) -> jnp.ndarray:
    occupied = state.black_bits | state.white_bits
    legal = ~occupied
    return jnp.where(state.terminated, jnp.zeros_like(legal), legal)


def step(state: GomokuState, action: jnp.ndarray):
    board_size = state.board.shape[0]
    num_actions = board_size * board_size
    action = jnp.int32(action)
    safe_action = jnp.clip(action, 0, num_actions - 1).astype(jnp.int32)

    legal = legal_action_mask(state)
    in_range = (action >= 0) & (action < num_actions)
    legal_at_action = jax.lax.cond(
        in_range,
        lambda a: legal[a],
        lambda _: jnp.bool_(False),
        operand=safe_action,
    )

    can_play = (~state.terminated) & legal_at_action
    illegal_move = (~state.terminated) & (~legal_at_action)

    row = safe_action // board_size
    col = safe_action % board_size

    board_after = jax.lax.cond(
        can_play,
        lambda b: b.at[row, col].set(state.to_play),
        lambda b: b,
        operand=state.board,
    )

    black_set = can_play & (state.to_play == 1)
    white_set = can_play & (state.to_play == -1)
    black_after = state.black_bits.at[safe_action].set(state.black_bits[safe_action] | black_set)
    white_after = state.white_bits.at[safe_action].set(state.white_bits[safe_action] | white_set)

    player_bits_after = jax.lax.cond(
        state.to_play == 1,
        lambda _: black_after,
        lambda _: white_after,
        operand=None,
    )
    win = jax.lax.cond(
        can_play,
        lambda _: _has_five_from_bits(player_bits_after, board_size),
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
    last_action = jnp.where(can_play, safe_action, state.last_action).astype(jnp.int32)

    next_state = GomokuState(
        board=board_after,
        black_bits=black_after,
        white_bits=white_after,
        to_play=next_to_play,
        last_action=last_action,
        num_moves=num_moves.astype(jnp.int32),
        terminated=terminated.astype(jnp.bool_),
        winner=winner,
    )
    return next_state, reward, next_state.terminated


def encode_state(state: GomokuState) -> jnp.ndarray:
    board_size = state.board.shape[0]
    num_actions = board_size * board_size

    mine_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.black_bits,
        lambda _: state.white_bits,
        operand=None,
    )
    opp_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.white_bits,
        lambda _: state.black_bits,
        operand=None,
    )
    mine = mine_bits.reshape(board_size, board_size).astype(jnp.float32)
    opp = opp_bits.reshape(board_size, board_size).astype(jnp.float32)

    safe_last_action = jnp.clip(state.last_action, 0, num_actions - 1)
    last_plane = jax.nn.one_hot(safe_last_action, num_actions, dtype=jnp.float32).reshape(board_size, board_size)
    last_plane = jnp.where(state.last_action >= 0, last_plane, jnp.zeros_like(last_plane))

    side_to_move = jnp.full((board_size, board_size), (state.to_play == 1).astype(jnp.float32), dtype=jnp.float32)
    return jnp.stack([mine, opp, last_plane, side_to_move], axis=-1)


legal_action_mask = jax.jit(legal_action_mask)
step = jax.jit(step)
encode_state = jax.jit(encode_state)

batch_legal_action_mask = jax.vmap(legal_action_mask)
batch_encode_states = jax.vmap(encode_state)
batch_step = jax.vmap(step)
