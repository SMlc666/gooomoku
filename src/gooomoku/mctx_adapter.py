from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
import mctx

from gooomoku import env
from gooomoku.net import PolicyValueNet


def _masked_logits(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(legal_mask, logits, min_logit)


@functools.lru_cache(maxsize=None)
def _line_window_indices_np(board_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    east_starts, south_starts, dr_starts, dl_starts = env._line_start_masks_np(board_size)

    def make_indices(starts: np.ndarray, offset: int) -> np.ndarray:
        start_idx = np.nonzero(starts)[0].astype(np.int32)
        if start_idx.size == 0:
            return np.zeros((0, 5), dtype=np.int32)
        offsets = np.arange(5, dtype=np.int32)[None, :] * np.int32(offset)
        return start_idx[:, None] + offsets

    return (
        make_indices(east_starts, 1),
        make_indices(south_starts, board_size),
        make_indices(dr_starts, board_size + 1),
        make_indices(dl_starts, board_size - 1),
    )


def _line_window_indices(board_size: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    east, south, dr, dl = _line_window_indices_np(board_size)
    return (
        jnp.asarray(east, dtype=jnp.int32),
        jnp.asarray(south, dtype=jnp.int32),
        jnp.asarray(dr, dtype=jnp.int32),
        jnp.asarray(dl, dtype=jnp.int32),
    )


def _segment_add(indices: jnp.ndarray, picks: jnp.ndarray, num_actions: int) -> jnp.ndarray:
    if indices.shape[0] == 0:
        return jnp.zeros((num_actions,), dtype=jnp.int32)
    return jnp.bincount(
        indices.reshape(-1),
        weights=picks.astype(jnp.int32).reshape(-1),
        length=num_actions,
    ).astype(jnp.int32)


def _winning_counts_from_windows(
    *,
    player_bits: jnp.ndarray,
    legal: jnp.ndarray,
    window_indices: jnp.ndarray,
    num_actions: int,
) -> jnp.ndarray:
    if window_indices.shape[0] == 0:
        return jnp.zeros((num_actions,), dtype=jnp.int32)
    stone_windows = player_bits[window_indices]
    empty_windows = legal[window_indices]
    winning_window = (jnp.sum(stone_windows.astype(jnp.int32), axis=1) == 4) & (
        jnp.sum(empty_windows.astype(jnp.int32), axis=1) == 1
    )
    winning_picks = winning_window[:, None] & empty_windows
    return _segment_add(window_indices, winning_picks, num_actions)


def _forcing_threat_counts_from_windows(
    *,
    player_bits: jnp.ndarray,
    legal: jnp.ndarray,
    window_indices: jnp.ndarray,
    num_actions: int,
) -> jnp.ndarray:
    if window_indices.shape[0] == 0:
        return jnp.zeros((num_actions,), dtype=jnp.int32)
    stone_windows = player_bits[window_indices]
    empty_windows = legal[window_indices]
    forcing_window = (jnp.sum(stone_windows.astype(jnp.int32), axis=1) == 3) & (
        jnp.sum(empty_windows.astype(jnp.int32), axis=1) == 2
    )
    forcing_picks = forcing_window[:, None] & empty_windows
    return _segment_add(window_indices, forcing_picks, num_actions)


def _winning_moves_for_bits(
    *,
    player_bits: jnp.ndarray,
    occupied: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    legal = ~occupied
    east_idx, south_idx, dr_idx, dl_idx = _line_window_indices(board_size)
    counts = jnp.zeros((num_actions,), dtype=jnp.int32)
    counts = counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=east_idx,
        num_actions=num_actions,
    )
    counts = counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=south_idx,
        num_actions=num_actions,
    )
    counts = counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dr_idx,
        num_actions=num_actions,
    )
    counts = counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dl_idx,
        num_actions=num_actions,
    )
    return legal & (counts > 0)


def _urgent_moves_for_player(
    *,
    player_bits: jnp.ndarray,
    occupied: jnp.ndarray,
    legal: jnp.ndarray,
    board_size: int,
    num_actions: int,
) -> jnp.ndarray:
    del occupied
    legal = legal.astype(jnp.bool_)

    east_idx, south_idx, dr_idx, dl_idx = _line_window_indices(board_size)

    win_counts = jnp.zeros((num_actions,), dtype=jnp.int32)
    win_counts = win_counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=east_idx,
        num_actions=num_actions,
    )
    win_counts = win_counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=south_idx,
        num_actions=num_actions,
    )
    win_counts = win_counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dr_idx,
        num_actions=num_actions,
    )
    win_counts = win_counts + _winning_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dl_idx,
        num_actions=num_actions,
    )
    win_mask = legal & (win_counts > 0)

    forcing_counts = jnp.zeros((num_actions,), dtype=jnp.int32)
    forcing_counts = forcing_counts + _forcing_threat_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=east_idx,
        num_actions=num_actions,
    )
    forcing_counts = forcing_counts + _forcing_threat_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=south_idx,
        num_actions=num_actions,
    )
    forcing_counts = forcing_counts + _forcing_threat_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dr_idx,
        num_actions=num_actions,
    )
    forcing_counts = forcing_counts + _forcing_threat_counts_from_windows(
        player_bits=player_bits,
        legal=legal,
        window_indices=dl_idx,
        num_actions=num_actions,
    )
    forcing_mask = legal & (forcing_counts > 0)

    num_winning_actions = jnp.sum(win_mask.astype(jnp.int32))
    remaining_win_after_move = (num_winning_actions - win_mask.astype(jnp.int32)) > 0
    return legal & (win_mask | remaining_win_after_move | forcing_mask)


def _current_player_winning_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = int(state.black_bits.shape[0])
    my_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.black_bits,
        lambda _: state.white_bits,
        operand=None,
    )
    occupied = state.black_bits | state.white_bits
    return _winning_moves_for_bits(
        player_bits=my_bits,
        occupied=occupied,
        board_size=board_size,
        num_actions=num_actions,
    )


def _opponent_threat_block_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = int(state.black_bits.shape[0])

    opp_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.white_bits,
        lambda _: state.black_bits,
        operand=None,
    )
    occupied = state.black_bits | state.white_bits
    legal = env.legal_action_mask(state)
    # Block all opponent next moves that would create immediate win or rush-four.
    return _urgent_moves_for_player(
        player_bits=opp_bits,
        occupied=occupied,
        legal=legal,
        board_size=board_size,
        num_actions=num_actions,
    )


def _current_player_urgent_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = int(state.black_bits.shape[0])
    my_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.black_bits,
        lambda _: state.white_bits,
        operand=None,
    )
    occupied = state.black_bits | state.white_bits
    legal = env.legal_action_mask(state)
    return _urgent_moves_for_player(
        player_bits=my_bits,
        occupied=occupied,
        legal=legal,
        board_size=board_size,
        num_actions=num_actions,
    )


def _opponent_immediate_winning_mask(state: env.GomokuState) -> jnp.ndarray:
    board_size = int(state.board.shape[0])
    num_actions = int(state.black_bits.shape[0])
    opp_bits = jax.lax.cond(
        state.to_play == 1,
        lambda _: state.white_bits,
        lambda _: state.black_bits,
        operand=None,
    )
    occupied = state.black_bits | state.white_bits
    return _winning_moves_for_bits(
        player_bits=opp_bits,
        occupied=occupied,
        board_size=board_size,
        num_actions=num_actions,
    )


def _force_defense_prior_logits(prior_logits: jnp.ndarray, states: env.GomokuState) -> jnp.ndarray:
    win_mask = jax.vmap(_current_player_winning_mask)(states)
    has_win = jnp.any(win_mask, axis=-1, keepdims=True)
    win_probs = win_mask.astype(prior_logits.dtype)
    win_probs = win_probs / jnp.clip(jnp.sum(win_probs, axis=-1, keepdims=True), 1e-8)
    min_logit = jnp.finfo(prior_logits.dtype).min
    forced_win_logits = jnp.where(win_mask, jnp.log(jnp.clip(win_probs, 1e-20)), min_logit)

    my_urgent_mask = jax.vmap(_current_player_urgent_mask)(states)
    has_my_urgent = jnp.any(my_urgent_mask, axis=-1, keepdims=True)
    my_urgent_probs = my_urgent_mask.astype(prior_logits.dtype)
    my_urgent_probs = my_urgent_probs / jnp.clip(jnp.sum(my_urgent_probs, axis=-1, keepdims=True), 1e-8)
    forced_my_urgent_logits = jnp.where(my_urgent_mask, jnp.log(jnp.clip(my_urgent_probs, 1e-20)), min_logit)

    opp_immediate_mask = jax.vmap(_opponent_immediate_winning_mask)(states)
    has_opp_immediate = jnp.any(opp_immediate_mask, axis=-1, keepdims=True)
    opp_immediate_probs = opp_immediate_mask.astype(prior_logits.dtype)
    opp_immediate_probs = opp_immediate_probs / jnp.clip(jnp.sum(opp_immediate_probs, axis=-1, keepdims=True), 1e-8)
    forced_opp_immediate_logits = jnp.where(
        opp_immediate_mask,
        jnp.log(jnp.clip(opp_immediate_probs, 1e-20)),
        min_logit,
    )

    block_mask = jax.vmap(_opponent_threat_block_mask)(states)
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
):
    root = root_output(
        params=params,
        model=model,
        states=states,
        force_defense_at_root=force_defense_at_root,
    )
    invalid_actions = ~env.batch_legal_action_mask(states)
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
):
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
        if use_root_noise:
            noise_key, search_key = jax.random.split(rng_key)
            legal_actions = ~invalid_actions
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
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
        )

    return search_fn
