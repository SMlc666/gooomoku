from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

RULE_FORBID_BLACK_OVERLINE = np.int8(1)
RULE_FORBID_BLACK_DOUBLE_THREE = np.int8(2)
RULE_FORBID_BLACK_DOUBLE_FOUR = np.int8(4)
RULE_RENJU_FULL = np.int8(
    int(RULE_FORBID_BLACK_OVERLINE) | int(RULE_FORBID_BLACK_DOUBLE_THREE) | int(RULE_FORBID_BLACK_DOUBLE_FOUR)
)
OBS_PLANES = 7


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
    rule_flags: jnp.ndarray
    swap_source_flag: jnp.ndarray
    swap_applied_flag: jnp.ndarray


@functools.lru_cache(maxsize=None)
def _line_start_masks_np(board_size: int, line_len: int = 5):
    if line_len < 2:
        raise ValueError(f"line_len must be >= 2, got {line_len}")
    num_actions = board_size * board_size
    east = np.zeros((num_actions,), dtype=np.bool_)
    south = np.zeros((num_actions,), dtype=np.bool_)
    diag_dr = np.zeros((num_actions,), dtype=np.bool_)
    diag_dl = np.zeros((num_actions,), dtype=np.bool_)

    if board_size >= line_len:
        idx = np.arange(num_actions, dtype=np.int32).reshape(board_size, board_size)
        tail = line_len - 1
        east[idx[:, : board_size - tail].reshape(-1)] = True
        south[idx[: board_size - tail, :].reshape(-1)] = True
        diag_dr[idx[: board_size - tail, : board_size - tail].reshape(-1)] = True
        diag_dl[idx[: board_size - tail, tail:].reshape(-1)] = True

    return east, south, diag_dr, diag_dl


def _line_start_masks(board_size: int, line_len: int = 5):
    east, south, diag_dr, diag_dl = _line_start_masks_np(board_size, line_len)
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


def _has_line(bits: jnp.ndarray, starts: jnp.ndarray, offset: int, line_len: int = 5) -> jnp.ndarray:
    acc = starts
    for i in range(line_len):
        acc = acc & _shift_bits(bits, i * offset)
    return jnp.any(acc)


def _has_five_from_bits(bits: jnp.ndarray, board_size: int) -> jnp.ndarray:
    east_starts, south_starts, dr_starts, dl_starts = _line_start_masks(board_size, line_len=5)
    east = _has_line(bits, east_starts, 1, line_len=5)
    south = _has_line(bits, south_starts, board_size, line_len=5)
    diag_dr = _has_line(bits, dr_starts, board_size + 1, line_len=5)
    diag_dl = _has_line(bits, dl_starts, board_size - 1, line_len=5)
    return east | south | diag_dr | diag_dl


def _has_six_from_bits(bits: jnp.ndarray, board_size: int) -> jnp.ndarray:
    east_starts, south_starts, dr_starts, dl_starts = _line_start_masks(board_size, line_len=6)
    east = _has_line(bits, east_starts, 1, line_len=6)
    south = _has_line(bits, south_starts, board_size, line_len=6)
    diag_dr = _has_line(bits, dr_starts, board_size + 1, line_len=6)
    diag_dl = _has_line(bits, dl_starts, board_size - 1, line_len=6)
    return east | south | diag_dr | diag_dl


def _line_values_around_move(
    board: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    *,
    dr: int,
    dc: int,
    radius: int = 5,
) -> jnp.ndarray:
    board_size = board.shape[0]
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    rr = row + offsets * jnp.int32(dr)
    cc = col + offsets * jnp.int32(dc)
    in_bounds = (rr >= 0) & (rr < board_size) & (cc >= 0) & (cc < board_size)
    rr_clip = jnp.clip(rr, 0, board_size - 1)
    cc_clip = jnp.clip(cc, 0, board_size - 1)
    vals = board[rr_clip, cc_clip]
    return jnp.where(in_bounds, vals, jnp.int8(-1))


def _direction_has_overline(line: jnp.ndarray) -> jnp.ndarray:
    # line has length 11, center at idx=5.
    found = jnp.bool_(False)
    for start in range(0, 6):
        seg = line[start : start + 6]
        found = found | jnp.all(seg == jnp.int8(1))
    return found


def _direction_has_four(line: jnp.ndarray) -> jnp.ndarray:
    # Any 5-window through center with exactly one empty and no white/outside.
    has_four = jnp.bool_(False)
    for start in range(1, 6):
        seg = line[start : start + 5]
        black_n = jnp.sum((seg == jnp.int8(1)).astype(jnp.int32))
        empty_n = jnp.sum((seg == jnp.int8(0)).astype(jnp.int32))
        blocked_n = jnp.sum((seg == jnp.int8(-1)).astype(jnp.int32))
        left = line[start - 1]
        right = line[start + 5]
        exact_five_if_fill = (left != jnp.int8(1)) & (right != jnp.int8(1))
        has_four = has_four | ((black_n == 4) & (empty_n == 1) & (blocked_n == 0) & exact_five_if_fill)
    return has_four


def _direction_has_open_three(line: jnp.ndarray) -> jnp.ndarray:
    # Open-three approximation aligned with Renju intent:
    # 6-window through center: 0 [4-cells with 3 black + 1 empty] 0
    has_open_three = jnp.bool_(False)
    for start in range(1, 5):
        seg6 = line[start : start + 6]
        left_open = seg6[0] == jnp.int8(0)
        right_open = seg6[5] == jnp.int8(0)
        inner = seg6[1:5]
        black_n = jnp.sum((inner == jnp.int8(1)).astype(jnp.int32))
        empty_n = jnp.sum((inner == jnp.int8(0)).astype(jnp.int32))
        blocked_n = jnp.sum((inner == jnp.int8(-1)).astype(jnp.int32))
        has_open_three = has_open_three | (left_open & right_open & (black_n == 3) & (empty_n == 1) & (blocked_n == 0))
    return has_open_three


def _is_black_renju_forbidden(
    board_after: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    rule_flags: jnp.ndarray,
) -> jnp.ndarray:
    renju_enabled = rule_flags != jnp.int8(0)
    if_not_enabled = jnp.bool_(False)

    def _compute(_):
        overline = jnp.bool_(False)
        four_dirs = jnp.int32(0)
        open_three_dirs = jnp.int32(0)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = _line_values_around_move(board_after, row, col, dr=dr, dc=dc, radius=5)
            overline = overline | _direction_has_overline(line)
            four_dirs = four_dirs + _direction_has_four(line).astype(jnp.int32)
            open_three_dirs = open_three_dirs + _direction_has_open_three(line).astype(jnp.int32)

        forbid_overline = ((rule_flags & RULE_FORBID_BLACK_OVERLINE) != 0) & overline
        forbid_double_four = ((rule_flags & RULE_FORBID_BLACK_DOUBLE_FOUR) != 0) & (four_dirs >= 2)
        forbid_double_three = ((rule_flags & RULE_FORBID_BLACK_DOUBLE_THREE) != 0) & (open_three_dirs >= 2)
        return forbid_overline | forbid_double_four | forbid_double_three

    return jax.lax.cond(renju_enabled, _compute, lambda _: if_not_enabled, operand=None)


def reset(
    board_size: int,
    *,
    rule_flags: int = 0,
    swap_source_flag: int = 0,
    swap_applied_flag: int = 0,
) -> GomokuState:
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
        rule_flags=jnp.int8(rule_flags),
        swap_source_flag=jnp.int8(swap_source_flag),
        swap_applied_flag=jnp.int8(swap_applied_flag),
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

    row = safe_action // board_size
    col = safe_action % board_size
    can_play_proposed = (~state.terminated) & legal_at_action

    black_set_proposed = can_play_proposed & (state.to_play == 1)
    board_after_proposed = jax.lax.cond(
        can_play_proposed,
        lambda b: b.at[row, col].set(state.to_play),
        lambda b: b,
        operand=state.board,
    )
    forbidden_black_move = jax.lax.cond(
        can_play_proposed & (state.to_play == 1),
        lambda _: _is_black_renju_forbidden(board_after_proposed, row, col, state.rule_flags),
        lambda _: jnp.bool_(False),
        operand=None,
    )
    can_play = can_play_proposed & (~forbidden_black_move)
    illegal_move = (~state.terminated) & ((~legal_at_action) | forbidden_black_move)

    board_after = jax.lax.cond(can_play, lambda b: board_after_proposed, lambda b: b, operand=state.board)

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
        rule_flags=state.rule_flags,
        swap_source_flag=state.swap_source_flag,
        swap_applied_flag=state.swap_applied_flag,
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
    forbid_black = jnp.full(
        (board_size, board_size),
        (state.rule_flags != jnp.int8(0)).astype(jnp.float32),
        dtype=jnp.float32,
    )
    swap_source = jnp.full(
        (board_size, board_size),
        (state.swap_source_flag != 0).astype(jnp.float32),
        dtype=jnp.float32,
    )
    swap_applied = jnp.full(
        (board_size, board_size),
        (state.swap_applied_flag != 0).astype(jnp.float32),
        dtype=jnp.float32,
    )
    return jnp.stack([mine, opp, last_plane, side_to_move, forbid_black, swap_source, swap_applied], axis=-1)


legal_action_mask = jax.jit(legal_action_mask)
step = jax.jit(step)
encode_state = jax.jit(encode_state)

batch_legal_action_mask = jax.vmap(legal_action_mask)
batch_encode_states = jax.vmap(encode_state)
batch_step = jax.vmap(step)
