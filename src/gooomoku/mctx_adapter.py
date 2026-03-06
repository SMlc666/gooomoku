from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
import mctx

from gooomoku.cpp_backend import CppSearchEngine
from gooomoku.cpp_backend import numpy_dict_to_gomoku_state
from gooomoku.cpp_backend import is_cpp_backend_available
from gooomoku import env
from gooomoku.net import PolicyValueNet


def _masked_logits(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(legal_mask, logits, min_logit)


@functools.lru_cache(maxsize=None)
def _line_window_action_indices_np(board_size: int) -> np.ndarray:
    east_starts, south_starts, dr_starts, dl_starts = env._line_start_masks_np(board_size)

    def make_action_idx(starts: np.ndarray, offset: int) -> np.ndarray:
        start_idx = np.nonzero(starts)[0].astype(np.int32)
        if start_idx.size == 0:
            return np.zeros((0, 5), dtype=np.int32)
        offsets = np.arange(5, dtype=np.int32)[None, :] * np.int32(offset)
        return start_idx[:, None] + offsets

    east = make_action_idx(east_starts, 1)
    south = make_action_idx(south_starts, board_size)
    diag_dr = make_action_idx(dr_starts, board_size + 1)
    diag_dl = make_action_idx(dl_starts, board_size - 1)
    return np.concatenate((east, south, diag_dr, diag_dl), axis=0)


@functools.lru_cache(maxsize=None)
def _line_window_membership_matrix_np(board_size: int) -> np.ndarray:
    action_idx = _line_window_action_indices_np(board_size)
    num_actions = board_size * board_size
    num_windows = action_idx.shape[0]
    if num_windows == 0:
        return np.zeros((0, num_actions), dtype=np.float32)
    matrix = np.zeros((num_windows, num_actions), dtype=np.float32)
    matrix[np.arange(num_windows, dtype=np.int32)[:, None], action_idx] = np.float32(1.0)
    return matrix


def _line_window_membership_matrix(board_size: int) -> jnp.ndarray:
    return jnp.asarray(_line_window_membership_matrix_np(board_size), dtype=jnp.float32)


def _window_pattern_counts(
    *,
    player_bits: jnp.ndarray,
    legal_bits: jnp.ndarray,
    window_membership_matrix: jnp.ndarray,
    target_stones: int,
    target_empties: int,
    num_actions: int,
) -> jnp.ndarray:
    if window_membership_matrix.shape[0] == 0:
        return jnp.zeros((num_actions,), dtype=jnp.int32)
    player_bits_f = player_bits.astype(jnp.float32)
    legal_bits_f = legal_bits.astype(jnp.float32)
    stone_counts = jnp.matmul(window_membership_matrix, player_bits_f)
    empty_counts = jnp.matmul(window_membership_matrix, legal_bits_f)
    matches = (stone_counts == jnp.float32(target_stones)) & (empty_counts == jnp.float32(target_empties))
    counts = jnp.matmul(matches.astype(jnp.float32), window_membership_matrix)
    counts = counts * legal_bits_f
    return counts.astype(jnp.int32)


def _window_pattern_counts_batched(
    *,
    player_bits: jnp.ndarray,
    legal_bits: jnp.ndarray,
    window_membership_matrix: jnp.ndarray,
    target_stones: int,
    target_empties: int,
    num_actions: int,
) -> jnp.ndarray:
    if window_membership_matrix.shape[0] == 0:
        return jnp.zeros((player_bits.shape[0], num_actions), dtype=jnp.int32)
    player_bits_f = player_bits.astype(jnp.float32)
    legal_bits_f = legal_bits.astype(jnp.float32)
    stone_counts = jnp.matmul(player_bits_f, window_membership_matrix.T)
    empty_counts = jnp.matmul(legal_bits_f, window_membership_matrix.T)
    matches = (stone_counts == jnp.float32(target_stones)) & (empty_counts == jnp.float32(target_empties))
    counts = jnp.matmul(matches.astype(jnp.float32), window_membership_matrix)
    counts = counts * legal_bits_f
    return counts.astype(jnp.int32)


def _winning_moves_for_words(
    *,
    player_words: jnp.ndarray,
    legal_bits: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    player_bits = env.unpack_bits(player_words, num_actions=num_actions)
    window_membership_matrix = _line_window_membership_matrix(board_size)
    counts = _window_pattern_counts(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=4,
        target_empties=1,
        num_actions=num_actions,
    )
    return legal_bits & (counts > 0)


def _urgent_moves_for_words(
    *,
    player_words: jnp.ndarray,
    legal_bits: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    player_bits = env.unpack_bits(player_words, num_actions=num_actions)
    window_membership_matrix = _line_window_membership_matrix(board_size)

    win_counts = _window_pattern_counts(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=4,
        target_empties=1,
        num_actions=num_actions,
    )
    win_mask = legal_bits & (win_counts > 0)

    forcing_counts = _window_pattern_counts(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=3,
        target_empties=2,
        num_actions=num_actions,
    )
    forcing_mask = legal_bits & (forcing_counts > 0)

    num_winning_actions = jnp.sum(win_mask.astype(jnp.int32))
    remaining_win_after_move = (num_winning_actions - win_mask.astype(jnp.int32)) > 0
    return legal_bits & (win_mask | remaining_win_after_move | forcing_mask)


def _winning_moves_for_words_batched(
    *,
    player_words: jnp.ndarray,
    legal_bits: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    player_bits = env.unpack_bits(player_words, num_actions=num_actions)
    window_membership_matrix = _line_window_membership_matrix(board_size)
    counts = _window_pattern_counts_batched(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=4,
        target_empties=1,
        num_actions=num_actions,
    )
    return legal_bits & (counts > 0)


def _urgent_moves_for_words_batched(
    *,
    player_words: jnp.ndarray,
    legal_bits: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    player_bits = env.unpack_bits(player_words, num_actions=num_actions)
    window_membership_matrix = _line_window_membership_matrix(board_size)

    win_counts = _window_pattern_counts_batched(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=4,
        target_empties=1,
        num_actions=num_actions,
    )
    win_mask = legal_bits & (win_counts > 0)

    forcing_counts = _window_pattern_counts_batched(
        player_bits=player_bits,
        legal_bits=legal_bits,
        window_membership_matrix=window_membership_matrix,
        target_stones=3,
        target_empties=2,
        num_actions=num_actions,
    )
    forcing_mask = legal_bits & (forcing_counts > 0)

    num_winning_actions = jnp.sum(win_mask.astype(jnp.int32), axis=1, keepdims=True)
    remaining_win_after_move = (num_winning_actions - win_mask.astype(jnp.int32)) > 0
    return legal_bits & (win_mask | remaining_win_after_move | forcing_mask)


def _batched_force_masks(states: env.GomokuState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    board_size = int(states.board.shape[1])
    num_actions = board_size * board_size
    to_play_black = states.to_play == 1
    my_words = jnp.where(to_play_black[:, None], states.black_words, states.white_words)
    opp_words = jnp.where(to_play_black[:, None], states.white_words, states.black_words)
    occupied_words = jnp.bitwise_or(states.black_words, states.white_words)
    legal_bits = env.unpack_bits(jnp.bitwise_not(occupied_words), num_actions=num_actions)
    legal_bits = jnp.where(states.terminated[:, None], jnp.zeros_like(legal_bits), legal_bits)

    win_mask = _winning_moves_for_words_batched(
        player_words=my_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )
    my_urgent_mask = _urgent_moves_for_words_batched(
        player_words=my_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )
    opp_immediate_mask = _winning_moves_for_words_batched(
        player_words=opp_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )
    block_mask = _urgent_moves_for_words_batched(
        player_words=opp_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )
    return win_mask, my_urgent_mask, opp_immediate_mask, block_mask


def _legal_bits_from_words(occupied_words: jnp.ndarray, *, num_actions: int, terminated: jnp.ndarray) -> jnp.ndarray:
    legal = env.unpack_bits(jnp.bitwise_not(occupied_words), num_actions=num_actions)
    return jnp.where(terminated, jnp.zeros_like(legal), legal)


def _current_player_winning_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = board_size * board_size
    my_words = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.black_words,
        lambda _: state.white_words,
        operand=None,
    )
    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    legal_bits = _legal_bits_from_words(occupied_words, num_actions=num_actions, terminated=state.terminated)
    return _winning_moves_for_words(
        player_words=my_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )


def _opponent_threat_block_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = board_size * board_size
    opp_words = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.white_words,
        lambda _: state.black_words,
        operand=None,
    )
    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    legal_bits = _legal_bits_from_words(occupied_words, num_actions=num_actions, terminated=state.terminated)
    # Block all opponent next moves that would create immediate win or rush-four.
    return _urgent_moves_for_words(
        player_words=opp_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )


def _current_player_urgent_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = board_size * board_size
    my_words = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.black_words,
        lambda _: state.white_words,
        operand=None,
    )
    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    legal_bits = _legal_bits_from_words(occupied_words, num_actions=num_actions, terminated=state.terminated)
    return _urgent_moves_for_words(
        player_words=my_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )


def _opponent_immediate_winning_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = board_size * board_size
    opp_words = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.white_words,
        lambda _: state.black_words,
        operand=None,
    )
    occupied_words = jnp.bitwise_or(state.black_words, state.white_words)
    legal_bits = _legal_bits_from_words(occupied_words, num_actions=num_actions, terminated=state.terminated)
    return _winning_moves_for_words(
        player_words=opp_words,
        legal_bits=legal_bits,
        board_size=board_size,
        num_actions=num_actions,
    )


def _force_defense_prior_logits(prior_logits: jnp.ndarray, states: env.GomokuState) -> jnp.ndarray:
    win_mask, my_urgent_mask, opp_immediate_mask, block_mask = _batched_force_masks(states)
    has_win = jnp.any(win_mask, axis=-1, keepdims=True)
    win_probs = win_mask.astype(prior_logits.dtype)
    win_probs = win_probs / jnp.clip(jnp.sum(win_probs, axis=-1, keepdims=True), 1e-8)
    min_logit = jnp.finfo(prior_logits.dtype).min
    forced_win_logits = jnp.where(win_mask, jnp.log(jnp.clip(win_probs, 1e-20)), min_logit)

    has_my_urgent = jnp.any(my_urgent_mask, axis=-1, keepdims=True)
    my_urgent_probs = my_urgent_mask.astype(prior_logits.dtype)
    my_urgent_probs = my_urgent_probs / jnp.clip(jnp.sum(my_urgent_probs, axis=-1, keepdims=True), 1e-8)
    forced_my_urgent_logits = jnp.where(my_urgent_mask, jnp.log(jnp.clip(my_urgent_probs, 1e-20)), min_logit)

    has_opp_immediate = jnp.any(opp_immediate_mask, axis=-1, keepdims=True)
    opp_immediate_probs = opp_immediate_mask.astype(prior_logits.dtype)
    opp_immediate_probs = opp_immediate_probs / jnp.clip(jnp.sum(opp_immediate_probs, axis=-1, keepdims=True), 1e-8)
    forced_opp_immediate_logits = jnp.where(
        opp_immediate_mask,
        jnp.log(jnp.clip(opp_immediate_probs, 1e-20)),
        min_logit,
    )

    has_threat = jnp.any(block_mask, axis=-1, keepdims=True)

    block_probs = block_mask.astype(prior_logits.dtype)
    block_probs = block_probs / jnp.clip(jnp.sum(block_probs, axis=-1, keepdims=True), 1e-8)
    forced_logits = jnp.where(block_mask, jnp.log(jnp.clip(block_probs, 1e-20)), min_logit)
    with_defense = jnp.where(has_threat, forced_logits, prior_logits)
    with_my_urgent = jnp.where(has_my_urgent, forced_my_urgent_logits, with_defense)
    with_opp_immediate = jnp.where(has_opp_immediate, forced_opp_immediate_logits, with_my_urgent)
    return jnp.where(has_win, forced_win_logits, with_opp_immediate)


def _apply_root_dirichlet_noise(
    prior_logits: jnp.ndarray,
    legal_mask: jnp.ndarray,
    rng_key,
    fraction: float,
    alpha: float,
) -> jnp.ndarray:
    probs = jax.nn.softmax(prior_logits, axis=-1)
    num_actions = prior_logits.shape[-1]

    noise = jax.random.dirichlet(
        rng_key,
        jnp.full((num_actions,), jnp.float32(alpha), dtype=probs.dtype),
        shape=(prior_logits.shape[0],),
    )
    legal = legal_mask.astype(probs.dtype)
    noise = noise * legal
    noise = noise / jnp.clip(jnp.sum(noise, axis=-1, keepdims=True), 1e-8)

    mixed = (jnp.float32(1.0) - jnp.float32(fraction)) * probs + jnp.float32(fraction) * noise
    mixed = mixed * legal
    mixed = mixed / jnp.clip(jnp.sum(mixed, axis=-1, keepdims=True), 1e-8)

    min_logit = jnp.finfo(prior_logits.dtype).min
    return jnp.where(legal_mask, jnp.log(jnp.clip(mixed, 1e-20)), min_logit)


def _apply_dynamic_considered_pruning(
    *,
    prior_logits: jnp.ndarray,
    legal_mask: jnp.ndarray,
    num_moves: jnp.ndarray,
    max_num_considered_actions: int,
    opening_considered_actions: int,
    midgame_considered_actions: int,
    endgame_considered_actions: int,
    midgame_start_move: int,
    endgame_start_move: int,
) -> jnp.ndarray:
    num_actions = prior_logits.shape[-1]
    top_k = int(max(1, min(max_num_considered_actions, num_actions)))
    _, top_idx = jax.lax.top_k(prior_logits, k=top_k)

    opening_k = int(max(1, min(opening_considered_actions, top_k)))
    mid_k = int(max(1, min(midgame_considered_actions, top_k)))
    end_k = int(max(1, min(endgame_considered_actions, top_k)))
    move_count = num_moves.astype(jnp.int32)
    desired_k = jnp.where(
        move_count < jnp.int32(midgame_start_move),
        jnp.int32(opening_k),
        jnp.where(move_count < jnp.int32(endgame_start_move), jnp.int32(mid_k), jnp.int32(end_k)),
    )
    keep_rank = jnp.arange(top_k, dtype=jnp.int32)[None, :] < desired_k[:, None]

    batch_size = prior_logits.shape[0]
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    selected = jnp.zeros((batch_size, num_actions), dtype=jnp.bool_)
    selected = selected.at[batch_idx, top_idx].set(keep_rank)
    selected = selected & legal_mask

    min_logit = jnp.finfo(prior_logits.dtype).min
    return jnp.where(selected, prior_logits, min_logit)


def root_output(
    params,
    model: PolicyValueNet,
    states: env.GomokuState,
    *,
    force_defense_at_root: bool = False,
) -> mctx.RootFnOutput:
    obs = env.batch_encode_states(states)
    logits, value = model.apply(params, obs)
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    legal = env.batch_legal_action_mask(states)
    prior_logits = _masked_logits(logits, legal)
    if force_defense_at_root:
        prior_logits = _force_defense_prior_logits(prior_logits, states)
    return mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=states)


def recurrent_fn(
    params,
    rng_key,
    action,
    embedding: env.GomokuState,
    *,
    model: PolicyValueNet,
    force_defense_in_recurrent: bool = False,
):
    del rng_key
    next_state, reward, done = env.batch_step(embedding, action)
    obs = env.batch_encode_states(next_state)
    logits, value = model.apply(params, obs)
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    legal = env.batch_legal_action_mask(next_state)
    prior_logits = _masked_logits(logits, legal)
    if force_defense_in_recurrent:
        prior_logits = _force_defense_prior_logits(prior_logits, next_state)
    discount = jnp.where(done, jnp.float32(0.0), jnp.float32(-1.0))
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return output, next_state


def run_gumbel_search(
    *,
    params,
    model: PolicyValueNet,
    states: env.GomokuState,
    rng_key,
    num_simulations: int,
    max_num_considered_actions: int,
    gumbel_scale: float = 1.0,
    force_defense_at_root: bool = False,
    force_defense_in_recurrent: bool = False,
    c_lcb: float = 0.0,
):
    root = root_output(
        params=params,
        model=model,
        states=states,
        force_defense_at_root=force_defense_at_root,
    )
    invalid_actions = ~env.batch_legal_action_mask(states)
    if c_lcb > 0.0:
        def qtransform(tree, node_index):
            base_qvalues = mctx.qtransform_completed_by_mix_value(tree, node_index)
            visit_counts = tree.children_visits[node_index]
            node_visit = tree.node_visits[node_index]
            lcb_penalty = c_lcb * jnp.sqrt(jnp.maximum(1.0, node_visit)) / (1 + visit_counts)
            return base_qvalues - lcb_penalty
    else:
        qtransform = mctx.qtransform_completed_by_mix_value

    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=functools.partial(
            recurrent_fn,
            model=model,
            force_defense_in_recurrent=force_defense_in_recurrent,
        ),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
        qtransform=qtransform,
    )


def build_search_fn(
    *,
    model: PolicyValueNet,
    num_simulations: int,
    max_num_considered_actions: int,
    gumbel_scale: float = 1.0,
    root_dirichlet_fraction: float = 0.0,
    root_dirichlet_alpha: float = 0.03,
    force_defense_at_root: bool = True,
    force_defense_in_recurrent: bool = False,
    c_lcb: float = 0.0,
    dynamic_considered_actions: bool = False,
    opening_considered_actions: int = 64,
    midgame_considered_actions: int = 96,
    endgame_considered_actions: int = 160,
    midgame_start_move: int = 12,
    endgame_start_move: int = 40,
    mcts_backend: str = "mctx",
    cpp_virtual_loss: float = 1.0,
    cpp_c_puct: float = 1.5,
    cpp_num_threads: int = 0,
    cpp_leaf_eval_batch_size: int = 512,
):
    backend = str(mcts_backend).strip().lower()
    if backend not in {"mctx", "cpp"}:
        raise ValueError(f"unsupported mcts_backend={mcts_backend!r}, expected 'mctx' or 'cpp'")

    if backend == "cpp":
        if force_defense_in_recurrent:
            raise ValueError("force_defense_in_recurrent is not supported by mcts_backend='cpp'")
        if not is_cpp_backend_available():
            raise RuntimeError(
                "mcts_backend='cpp' requested but gooomoku_cpp extension is unavailable. "
                "Build with: python setup.py build_ext --inplace"
            )
        cpp_engine = CppSearchEngine(
            model=model,
            board_size=int(model.board_size),
            leaf_eval_batch_size=cpp_leaf_eval_batch_size,
            num_threads=cpp_num_threads,
            virtual_loss=cpp_virtual_loss,
            c_puct=cpp_c_puct,
        )
        use_root_noise = root_dirichlet_fraction > 0.0

        def search_fn(params, states: env.GomokuState | dict, rng_key):
            search_key = rng_key
            states_for_cpp = states
            if isinstance(states, dict) and (not force_defense_at_root):
                state_np, prior_logits_np, value_np, legal_np = cpp_engine.root_policy(
                    params=params,
                    states=states,
                )
                states_for_cpp = state_np
                prior_logits = jnp.asarray(prior_logits_np, dtype=jnp.float32)
                root_value = jnp.asarray(value_np, dtype=jnp.float32)
                legal_actions = jnp.asarray(legal_np, dtype=jnp.bool_)
                num_moves = jnp.asarray(state_np["num_moves"], dtype=jnp.int32)
            else:
                states_jax = numpy_dict_to_gomoku_state(states) if isinstance(states, dict) else states
                root = root_output(
                    params=params,
                    model=model,
                    states=states_jax,
                    force_defense_at_root=force_defense_at_root,
                )
                invalid_actions = ~env.batch_legal_action_mask(states_jax)
                legal_actions = ~invalid_actions
                prior_logits = root.prior_logits
                root_value = root.value
                num_moves = states_jax.num_moves
            if dynamic_considered_actions:
                prior_logits = _apply_dynamic_considered_pruning(
                    prior_logits=prior_logits,
                    legal_mask=legal_actions,
                    num_moves=num_moves,
                    max_num_considered_actions=max_num_considered_actions,
                    opening_considered_actions=opening_considered_actions,
                    midgame_considered_actions=midgame_considered_actions,
                    endgame_considered_actions=endgame_considered_actions,
                    midgame_start_move=midgame_start_move,
                    endgame_start_move=endgame_start_move,
                )
            if use_root_noise:
                noise_key, search_key = jax.random.split(rng_key)
                prior_logits = _apply_root_dirichlet_noise(
                    prior_logits=prior_logits,
                    legal_mask=legal_actions,
                    rng_key=noise_key,
                    fraction=root_dirichlet_fraction,
                    alpha=root_dirichlet_alpha,
                )

            return cpp_engine.search(
                params=params,
                states=states_for_cpp,
                root_prior_logits=prior_logits,
                root_values=root_value,
                rng_key=search_key,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
                gumbel_scale=gumbel_scale,
                c_lcb=c_lcb,
            )

        setattr(search_fn, "_cpp_engine", cpp_engine)
        return search_fn

    recurrent = functools.partial(
        recurrent_fn,
        model=model,
        force_defense_in_recurrent=force_defense_in_recurrent,
    )
    use_root_noise = root_dirichlet_fraction > 0.0

    @jax.jit
    def search_fn(params, states: env.GomokuState, rng_key):
        search_key = rng_key
        root = root_output(
            params=params,
            model=model,
            states=states,
            force_defense_at_root=force_defense_at_root,
        )
        invalid_actions = ~env.batch_legal_action_mask(states)
        legal_actions = ~invalid_actions
        if dynamic_considered_actions:
            pruned_logits = _apply_dynamic_considered_pruning(
                prior_logits=root.prior_logits,
                legal_mask=legal_actions,
                num_moves=states.num_moves,
                max_num_considered_actions=max_num_considered_actions,
                opening_considered_actions=opening_considered_actions,
                midgame_considered_actions=midgame_considered_actions,
                endgame_considered_actions=endgame_considered_actions,
                midgame_start_move=midgame_start_move,
                endgame_start_move=endgame_start_move,
            )
            root = mctx.RootFnOutput(
                prior_logits=pruned_logits,
                value=root.value,
                embedding=root.embedding,
            )
        if use_root_noise:
            noise_key, search_key = jax.random.split(rng_key)
            noisy_logits = _apply_root_dirichlet_noise(
                prior_logits=root.prior_logits,
                legal_mask=legal_actions,
                rng_key=noise_key,
                fraction=root_dirichlet_fraction,
                alpha=root_dirichlet_alpha,
            )
            root = mctx.RootFnOutput(
                prior_logits=noisy_logits,
                value=root.value,
                embedding=root.embedding,
            )
        if c_lcb > 0.0:
            def qtransform(tree, node_index):
                base_qvalues = mctx.qtransform_completed_by_mix_value(tree, node_index)
                visit_counts = tree.children_visits[node_index]
                node_visit = tree.node_visits[node_index]
                lcb_penalty = c_lcb * jnp.sqrt(jnp.maximum(1.0, node_visit)) / (1 + visit_counts)
                return base_qvalues - lcb_penalty
        else:
            qtransform = mctx.qtransform_completed_by_mix_value

        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
            qtransform=qtransform,
        )

    return search_fn
