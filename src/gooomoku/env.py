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

_WORD_BITS = 32
_WORD_DTYPE = jnp.uint32


@struct.dataclass
class GomokuState:
    board: jnp.ndarray
    black_words: jnp.ndarray
    white_words: jnp.ndarray
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


@functools.lru_cache(maxsize=None)
def _line_windows_np(board_size: int, line_len: int, offset: int):
    east, south, diag_dr, diag_dl = _line_start_masks_np(board_size, line_len)
    if offset == 1:
        starts = east
    elif offset == board_size:
        starts = south
    elif offset == board_size + 1:
        starts = diag_dr
    elif offset == board_size - 1:
        starts = diag_dl
    else:
        raise ValueError(f"unsupported offset for board_size={board_size}: {offset}")

    start_idx = np.flatnonzero(starts).astype(np.int32)
    if start_idx.size == 0:
        return np.zeros((0, line_len), dtype=np.int32)
    steps = np.arange(line_len, dtype=np.int32)[None, :]
    return start_idx[:, None] + steps * np.int32(offset)


@functools.lru_cache(maxsize=None)
def _packed_line_tables(board_size: int, line_len: int, offset: int):
    windows = _line_windows_np(board_size, line_len, offset)
    word_idx = windows // _WORD_BITS
    bit_idx = (windows % _WORD_BITS).astype(np.uint32)
    bit_masks = (np.uint32(1) << bit_idx).astype(np.uint32)
    return (
        jnp.asarray(word_idx, dtype=jnp.int32),
        jnp.asarray(bit_masks, dtype=_WORD_DTYPE),
    )


def _num_words_for_actions(num_actions: int) -> int:
    return (num_actions + _WORD_BITS - 1) // _WORD_BITS


def _state_num_actions(state: GomokuState) -> int:
    board_size = int(state.board.shape[-1])
    return board_size * board_size


def _expand_mask(mask: jnp.ndarray, target_ndim: int) -> jnp.ndarray:
    while mask.ndim < target_ndim:
        mask = mask[..., None]
    return mask


def pack_bits(bits: jnp.ndarray) -> jnp.ndarray:
    num_actions = int(bits.shape[-1])
    pad = (-num_actions) % _WORD_BITS
    padded = bits
    if pad:
        pad_width = [(0, 0)] * bits.ndim
        pad_width[-1] = (0, pad)
        padded = jnp.pad(bits, pad_width=pad_width, mode="constant", constant_values=False)

    words_shape = padded.shape[:-1] + (-1, _WORD_BITS)
    words_src = padded.reshape(words_shape).astype(_WORD_DTYPE)
    shifts = jnp.arange(_WORD_BITS, dtype=_WORD_DTYPE)
    masks = jnp.left_shift(jnp.ones((_WORD_BITS,), dtype=_WORD_DTYPE), shifts)
    return jnp.sum(words_src * masks, axis=-1, dtype=_WORD_DTYPE)


def unpack_bits(words: jnp.ndarray, *, num_actions: int) -> jnp.ndarray:
    idx = jnp.arange(num_actions, dtype=jnp.int32)
    word_idx = idx // _WORD_BITS
    bit_idx = (idx % _WORD_BITS).astype(_WORD_DTYPE)
    selected = jnp.take(words, word_idx, axis=-1)
    one = jnp.asarray(1, dtype=_WORD_DTYPE)
    return jnp.not_equal(jnp.bitwise_and(jnp.right_shift(selected, bit_idx), one), 0)


def black_bits(state: GomokuState) -> jnp.ndarray:
    return unpack_bits(state.black_words, num_actions=_state_num_actions(state))


def white_bits(state: GomokuState) -> jnp.ndarray:
    return unpack_bits(state.white_words, num_actions=_state_num_actions(state))


def color_bits(state: GomokuState) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_actions = _state_num_actions(state)
    return (
        unpack_bits(state.black_words, num_actions=num_actions),
        unpack_bits(state.white_words, num_actions=num_actions),
    )


def occupied_bits(state: GomokuState) -> jnp.ndarray:
    black, white = color_bits(state)
    return black | white


def current_player_bits(state: GomokuState) -> jnp.ndarray:
    black, white = color_bits(state)
    return jnp.where(_expand_mask(state.to_play == 1, black.ndim), black, white)


def opponent_bits(state: GomokuState) -> jnp.ndarray:
    black, white = color_bits(state)
    return jnp.where(_expand_mask(state.to_play == 1, black.ndim), white, black)


def player_view_bits(state: GomokuState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    black, white = color_bits(state)
    occupied = black | white
    to_play_black = _expand_mask(state.to_play == 1, black.ndim)
    my_bits = jnp.where(to_play_black, black, white)
    opp_bits = jnp.where(to_play_black, white, black)
    return my_bits, opp_bits, occupied


def _bit_is_set(words: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    word_idx = action // _WORD_BITS
    bit_idx = (action % _WORD_BITS).astype(_WORD_DTYPE)
    word = words[word_idx]
    one = jnp.asarray(1, dtype=_WORD_DTYPE)
    return jnp.not_equal(jnp.bitwise_and(jnp.right_shift(word, bit_idx), one), 0)


def _set_bit(words: jnp.ndarray, action: jnp.ndarray, enabled: jnp.ndarray) -> jnp.ndarray:
    word_idx = action // _WORD_BITS
    bit_idx = (action % _WORD_BITS).astype(_WORD_DTYPE)
    mask = jnp.left_shift(jnp.asarray(1, dtype=_WORD_DTYPE), bit_idx)
    current = words[word_idx]
    updated = jnp.where(enabled, jnp.bitwise_or(current, mask), current)
    return words.at[word_idx].set(updated)


def _shift_bits(bits: jnp.ndarray, offset: int) -> jnp.ndarray:
    if offset <= 0:
        return bits
    zeros = jnp.zeros((offset,), dtype=jnp.bool_)
    return jnp.concatenate([bits[offset:], zeros], axis=0)


def _has_line(bits: jnp.ndarray, starts: jnp.ndarray, offset: int, line_len: int = 5) -> jnp.ndarray:
    def body(i, acc):
        return acc & _shift_bits(bits, i * offset)

    acc = jax.lax.fori_loop(0, line_len, body, starts)
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


def _has_line_from_words(words: jnp.ndarray, board_size: int, *, offset: int, line_len: int) -> jnp.ndarray:
    word_idx, bit_masks = _packed_line_tables(board_size, line_len, offset)
    if word_idx.shape[0] == 0:
        return jnp.bool_(False)
    selected = words[word_idx]
    hits = jnp.not_equal(jnp.bitwise_and(selected, bit_masks), jnp.asarray(0, dtype=_WORD_DTYPE))
    return jnp.any(jnp.all(hits, axis=-1))


def _has_five_from_words(words: jnp.ndarray, board_size: int) -> jnp.ndarray:
    east = _has_line_from_words(words, board_size, offset=1, line_len=5)
    south = _has_line_from_words(words, board_size, offset=board_size, line_len=5)
    diag_dr = _has_line_from_words(words, board_size, offset=board_size + 1, line_len=5)
    diag_dl = _has_line_from_words(words, board_size, offset=board_size - 1, line_len=5)
    return east | south | diag_dr | diag_dl


def _has_six_from_words(words: jnp.ndarray, board_size: int) -> jnp.ndarray:
    east = _has_line_from_words(words, board_size, offset=1, line_len=6)
    south = _has_line_from_words(words, board_size, offset=board_size, line_len=6)
    diag_dr = _has_line_from_words(words, board_size, offset=board_size + 1, line_len=6)
    diag_dl = _has_line_from_words(words, board_size, offset=board_size - 1, line_len=6)
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
    starts = jnp.arange(0, 6, dtype=jnp.int32)[:, None]
    cols = jnp.arange(0, 6, dtype=jnp.int32)[None, :]
    windows = line[starts + cols]
    return jnp.any(jnp.all(windows == jnp.int8(1), axis=1))


def _direction_has_four(line: jnp.ndarray) -> jnp.ndarray:
    # Any 5-window through center with exactly one empty and no white/outside.
    starts = jnp.arange(1, 6, dtype=jnp.int32)
    cols = jnp.arange(0, 5, dtype=jnp.int32)[None, :]
    segments = line[starts[:, None] + cols]

    black_n = jnp.sum((segments == jnp.int8(1)).astype(jnp.int32), axis=1)
    empty_n = jnp.sum((segments == jnp.int8(0)).astype(jnp.int32), axis=1)
    blocked_n = jnp.sum((segments == jnp.int8(-1)).astype(jnp.int32), axis=1)
    left = line[starts - 1]
    right = line[starts + 5]
    exact_five_if_fill = (left != jnp.int8(1)) & (right != jnp.int8(1))

    match = (black_n == 4) & (empty_n == 1) & (blocked_n == 0) & exact_five_if_fill
    return jnp.any(match)


def _direction_has_open_three(line: jnp.ndarray) -> jnp.ndarray:
    # Open-three approximation aligned with Renju intent:
    # 6-window through center: 0 [4-cells with 3 black + 1 empty] 0
    starts = jnp.arange(1, 5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(0, 6, dtype=jnp.int32)[None, :]
    seg6 = line[starts + cols]

    left_open = seg6[:, 0] == jnp.int8(0)
    right_open = seg6[:, 5] == jnp.int8(0)
    inner = seg6[:, 1:5]
    black_n = jnp.sum((inner == jnp.int8(1)).astype(jnp.int32), axis=1)
    empty_n = jnp.sum((inner == jnp.int8(0)).astype(jnp.int32), axis=1)
    blocked_n = jnp.sum((inner == jnp.int8(-1)).astype(jnp.int32), axis=1)

    match = left_open & right_open & (black_n == 3) & (empty_n == 1) & (blocked_n == 0)
    return jnp.any(match)


def _is_black_renju_forbidden(
    board_after: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    rule_flags: jnp.ndarray,
) -> jnp.ndarray:
    renju_enabled = rule_flags != jnp.int8(0)
    if_not_enabled = jnp.bool_(False)

    def _compute(_):
        directions = jnp.asarray(((0, 1), (1, 0), (1, 1), (1, -1)), dtype=jnp.int32)
        lines = jax.vmap(
            lambda d: _line_values_around_move(
                board_after,
                row,
                col,
                dr=d[0],
                dc=d[1],
                radius=5,
            )
        )(directions)
        overline = jnp.any(jax.vmap(_direction_has_overline)(lines))
        four_dirs = jnp.sum(jax.vmap(_direction_has_four)(lines).astype(jnp.int32))
        open_three_dirs = jnp.sum(jax.vmap(_direction_has_open_three)(lines).astype(jnp.int32))

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
    num_words = _num_words_for_actions(num_actions)
    return GomokuState(
        board=jnp.zeros((board_size, board_size), dtype=jnp.int8),
        black_words=jnp.zeros((num_words,), dtype=_WORD_DTYPE),
        white_words=jnp.zeros((num_words,), dtype=_WORD_DTYPE),
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
    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    legal = ~unpack_bits(occupied_words, num_actions=_state_num_actions(state))
    return jnp.where(_expand_mask(state.terminated, legal.ndim), jnp.zeros_like(legal), legal)


def step(state: GomokuState, action: jnp.ndarray):
    board_size = state.board.shape[0]
    num_actions = board_size * board_size
    action = jnp.int32(action)
    safe_action = jnp.clip(action, 0, num_actions - 1).astype(jnp.int32)

    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    in_range = (action >= 0) & (action < num_actions)
    legal_at_action = jax.lax.cond(
        in_range,
        lambda a: ~_bit_is_set(occupied_words, a),
        lambda _: jnp.bool_(False),
        operand=safe_action,
    )

    row = safe_action // board_size
    col = safe_action % board_size
    can_play_proposed = (~state.terminated) & legal_at_action

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
    black_after = _set_bit(state.black_words, safe_action, black_set)
    white_after = _set_bit(state.white_words, safe_action, white_set)

    player_words_after = jax.lax.cond(
        state.to_play == 1,
        lambda _: black_after,
        lambda _: white_after,
        operand=None,
    )
    win = jax.lax.cond(
        can_play,
        lambda _: _has_five_from_words(player_words_after, board_size),
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
        black_words=black_after,
        white_words=white_after,
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

    mine_bits, opp_bits, _ = player_view_bits(state)
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
