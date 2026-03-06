from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku import env
from gooomoku.cpp_backend import CppSearchEngine
from gooomoku.cpp_backend import gomoku_state_to_numpy_dict
from gooomoku.cpp_backend import is_cpp_backend_available
from gooomoku.cpp_backend import rng_key_to_seed
from gooomoku.mctx_adapter import build_search_fn
from gooomoku.net import PolicyValueNet
from gooomoku.runtime import configure_jax_runtime


def _dtype_from_name(name: str):
    table = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[name]


@dataclass
class TrainingExample:
    observation: jnp.ndarray
    policy_target: jnp.ndarray
    value_target: float


def _add_batch_dim(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _build_mixed_init_state(
    *,
    board_size: int,
    num_games: int,
    rng_key: jnp.ndarray,
    mode_freestyle_weight: float,
    mode_forbidden_weight: float,
    mode_swap2_weight: float,
    mode_swap2_forbidden_weight: float,
    swap2_swap_prob: float,
) -> env.GomokuState:
    num_actions = board_size * board_size
    if num_actions < 3:
        raise ValueError("swap2 requires board_size*board_size >= 3")

    mode_weights = jnp.asarray(
        [
            max(0.0, float(mode_freestyle_weight)),
            max(0.0, float(mode_forbidden_weight)),
            max(0.0, float(mode_swap2_weight)),
            max(0.0, float(mode_swap2_forbidden_weight)),
        ],
        dtype=jnp.float32,
    )
    mode_total = jnp.sum(mode_weights)
    mode_probs = jnp.where(
        mode_total > 0.0,
        mode_weights / jnp.maximum(mode_total, 1e-12),
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    mode_logits = jnp.log(jnp.clip(mode_probs, 1e-12, 1.0))

    mode_key, opening_key, swap_key = jax.random.split(rng_key, 3)
    mode_ids = jax.random.categorical(mode_key, mode_logits, shape=(num_games,)).astype(jnp.int32)
    is_forbidden = (mode_ids == 1) | (mode_ids == 3)
    is_swap2 = (mode_ids == 2) | (mode_ids == 3)

    opening_scores = jax.random.uniform(opening_key, shape=(num_games, num_actions), dtype=jnp.float32)
    _, opening_idx = jax.lax.top_k(opening_scores, k=3)
    first = opening_idx[:, 0]
    second = opening_idx[:, 1]
    third = opening_idx[:, 2]
    batch_idx = jnp.arange(num_games, dtype=jnp.int32)

    black_bits = jnp.zeros((num_games, num_actions), dtype=jnp.bool_)
    white_bits = jnp.zeros((num_games, num_actions), dtype=jnp.bool_)
    black_bits = black_bits.at[batch_idx, first].set(is_swap2)
    white_bits = white_bits.at[batch_idx, second].set(is_swap2)
    black_bits = black_bits.at[batch_idx, third].set(is_swap2)

    board = jnp.zeros((num_games, board_size, board_size), dtype=jnp.int8)
    first_row, first_col = first // board_size, first % board_size
    second_row, second_col = second // board_size, second % board_size
    third_row, third_col = third // board_size, third % board_size
    board = board.at[batch_idx, first_row, first_col].set(jnp.where(is_swap2, jnp.int8(1), jnp.int8(0)))
    board = board.at[batch_idx, second_row, second_col].set(jnp.where(is_swap2, jnp.int8(-1), jnp.int8(0)))
    board = board.at[batch_idx, third_row, third_col].set(jnp.where(is_swap2, jnp.int8(1), jnp.int8(0)))

    to_play = jnp.where(is_swap2, jnp.int8(-1), jnp.int8(1))
    last_action = jnp.where(is_swap2, third.astype(jnp.int32), jnp.int32(-1))
    num_moves = jnp.where(is_swap2, jnp.int32(3), jnp.int32(0))
    terminated = jnp.zeros((num_games,), dtype=jnp.bool_)
    winner = jnp.zeros((num_games,), dtype=jnp.int8)
    rule_flags = jnp.where(is_forbidden, jnp.int8(env.RULE_RENJU_FULL), jnp.int8(0))

    swap_pick = jax.random.bernoulli(swap_key, p=jnp.float32(max(0.0, min(1.0, float(swap2_swap_prob)))), shape=(num_games,))
    swap_applied = is_swap2 & swap_pick
    black_before_swap = black_bits
    white_before_swap = white_bits
    black_bits = jnp.where(swap_applied[:, None], white_before_swap, black_before_swap)
    white_bits = jnp.where(swap_applied[:, None], black_before_swap, white_before_swap)
    board = jnp.where(swap_applied[:, None, None], -board, board)
    to_play = jnp.where(swap_applied, -to_play, to_play).astype(jnp.int8)

    return env.GomokuState(
        board=board,
        black_words=env.pack_bits(black_bits),
        white_words=env.pack_bits(white_bits),
        to_play=to_play,
        last_action=last_action.astype(jnp.int32),
        num_moves=num_moves.astype(jnp.int32),
        terminated=terminated,
        winner=winner,
        rule_flags=rule_flags.astype(jnp.int8),
        swap_source_flag=is_swap2.astype(jnp.int8),
        swap_applied_flag=swap_applied.astype(jnp.int8),
    )


def _sample_action_jax(visit_probs: jnp.ndarray, rng_key: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    def greedy(_):
        return jnp.argmax(visit_probs).astype(jnp.int32)

    def sample(_):
        safe_probs = jnp.maximum(visit_probs, 1e-8)
        logits = jnp.log(safe_probs) / temperature
        return jax.random.categorical(rng_key, logits).astype(jnp.int32)

    return jax.lax.cond(temperature <= 1e-6, greedy, sample, operand=None)


def _sample_actions_jax(visit_probs: jnp.ndarray, rng_key: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    def greedy(_):
        return jnp.argmax(visit_probs, axis=-1).astype(jnp.int32)

    def sample(_):
        safe_probs = jnp.maximum(visit_probs, 1e-8)
        logits = jnp.log(safe_probs) / temperature
        return jax.random.categorical(rng_key, logits, axis=-1).astype(jnp.int32)

    return jax.lax.cond(temperature <= 1e-6, greedy, sample, operand=None)


def _sample_actions_numpy(
    visit_probs: np.ndarray,
    *,
    temperature: float,
    active: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    batch = int(visit_probs.shape[0])
    actions = np.full((batch,), -1, dtype=np.int32)
    if temperature <= 1e-6:
        actions[active] = np.argmax(visit_probs[active], axis=-1).astype(np.int32)
        return actions

    for i in np.flatnonzero(active):
        probs = np.asarray(visit_probs[i], dtype=np.float64)
        probs = np.maximum(probs, 1e-12)
        logits = np.log(probs) / max(temperature, 1e-6)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)
        actions[i] = int(rng.choice(p.shape[0], p=p))
    return actions


def _build_play_many_games_fn_cpp(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    num_games: int,
    dynamic_considered_actions: bool,
    opening_considered_actions: int,
    midgame_considered_actions: int,
    endgame_considered_actions: int,
    midgame_start_move: int,
    endgame_start_move: int,
    temperature_drop_move: int,
    final_temperature: float,
    root_dirichlet_fraction: float,
    root_dirichlet_alpha: float,
    c_lcb: float,
    mode_freestyle_weight: float,
    mode_forbidden_weight: float,
    mode_swap2_weight: float,
    mode_swap2_forbidden_weight: float,
    swap2_swap_prob: float,
    gumbel_scale: float,
    mcts_backend: str,
    cpp_virtual_loss: float,
    cpp_c_puct: float,
    cpp_num_threads: int,
    cpp_leaf_eval_batch_size: int,
):
    if str(mcts_backend).strip().lower() != "cpp":
        raise ValueError("_build_play_many_games_fn_cpp requires mcts_backend='cpp'")
    if not is_cpp_backend_available():
        raise RuntimeError(
            "mcts_backend='cpp' requested but gooomoku_cpp extension is unavailable. "
            "Build with: python setup.py build_ext --inplace"
        )

    max_steps = board_size * board_size
    num_actions = max_steps
    search_fn = build_search_fn(
        model=model,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
        root_dirichlet_fraction=root_dirichlet_fraction,
        root_dirichlet_alpha=root_dirichlet_alpha,
        force_defense_at_root=False,
        c_lcb=c_lcb,
        dynamic_considered_actions=dynamic_considered_actions,
        opening_considered_actions=opening_considered_actions,
        midgame_considered_actions=midgame_considered_actions,
        endgame_considered_actions=endgame_considered_actions,
        midgame_start_move=midgame_start_move,
        endgame_start_move=endgame_start_move,
        mcts_backend="cpp",
        cpp_virtual_loss=cpp_virtual_loss,
        cpp_c_puct=cpp_c_puct,
        cpp_num_threads=cpp_num_threads,
        cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
    )
    board_ops = getattr(search_fn, "_cpp_engine", None)
    if board_ops is None:
        board_ops = CppSearchEngine(
            model=model,
            board_size=board_size,
            leaf_eval_batch_size=cpp_leaf_eval_batch_size,
            num_threads=cpp_num_threads,
            virtual_loss=cpp_virtual_loss,
            c_puct=cpp_c_puct,
        )

    def play_many_fn(params, rng_key: jnp.ndarray, temperature: jnp.ndarray):
        setup_seed = rng_key_to_seed(rng_key)
        setup_key = jax.random.PRNGKey(np.uint32(setup_seed & ((1 << 31) - 1)))
        batched_init_state = _build_mixed_init_state(
            board_size=board_size,
            num_games=num_games,
            rng_key=setup_key,
            mode_freestyle_weight=mode_freestyle_weight,
            mode_forbidden_weight=mode_forbidden_weight,
            mode_swap2_weight=mode_swap2_weight,
            mode_swap2_forbidden_weight=mode_swap2_forbidden_weight,
            swap2_swap_prob=swap2_swap_prob,
        )

        state = gomoku_state_to_numpy_dict(batched_init_state)
        obs_buf = np.zeros((num_games, max_steps, board_size, board_size, env.OBS_PLANES), dtype=np.float32)
        pi_buf = np.zeros((num_games, max_steps, num_actions), dtype=np.float32)
        to_play_buf = np.zeros((num_games, max_steps), dtype=np.int8)
        mask_buf = np.zeros((num_games, max_steps), dtype=np.bool_)
        action_buf = np.full((num_games, max_steps), -1, dtype=np.int32)
        steps = np.zeros((num_games,), dtype=np.int32)

        rollout_rng = np.random.default_rng(setup_seed + 0x9E3779B97F4A7C15)
        base_temperature = float(np.asarray(jax.device_get(temperature)).reshape(()))

        for step_idx in range(max_steps):
            active = ~state["terminated"]
            if not np.any(active):
                break

            obs = board_ops.batch_encode(state)
            policy_output = search_fn(params, state, jax.random.PRNGKey(np.uint32((setup_seed + step_idx + 1) & ((1 << 31) - 1))))
            visit_probs = np.asarray(jax.device_get(policy_output.action_weights), dtype=np.float32)

            move_temperature = base_temperature if step_idx < int(temperature_drop_move) else float(final_temperature)
            action = _sample_actions_numpy(
                visit_probs,
                temperature=move_temperature,
                active=active,
                rng=rollout_rng,
            )

            obs_buf[:, step_idx] = np.where(active[:, None, None, None], obs, 0.0)
            pi_buf[:, step_idx] = np.where(active[:, None], visit_probs, 0.0)
            to_play_buf[:, step_idx] = np.where(active, state["to_play"], 0).astype(np.int8)
            mask_buf[:, step_idx] = active
            action_buf[:, step_idx] = np.where(active, action, -1).astype(np.int32)
            steps = steps + active.astype(np.int32)

            state, _, _ = board_ops.batch_step(state, action)

        winners = np.asarray(state["winner"], dtype=np.int8)
        signed_values = np.where(
            winners[:, None] == 0,
            0.0,
            np.where(to_play_buf == winners[:, None], 1.0, -1.0),
        ).astype(np.float32)
        value_buf = np.where(mask_buf, signed_values, 0.0).astype(np.float32)
        pi_buf = np.where(mask_buf[:, :, None], pi_buf, 0.0).astype(np.float32)

        return (
            jnp.asarray(obs_buf),
            jnp.asarray(pi_buf),
            jnp.asarray(value_buf),
            jnp.asarray(mask_buf),
            jnp.asarray(action_buf),
            jnp.asarray(steps),
            jnp.asarray(winners),
        )

    return play_many_fn


def build_play_one_game_fn(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    dynamic_considered_actions: bool = False,
    opening_considered_actions: int = 64,
    midgame_considered_actions: int = 96,
    endgame_considered_actions: int = 160,
    midgame_start_move: int = 12,
    endgame_start_move: int = 40,
    temperature_drop_move: int = 12,
    final_temperature: float = 0.0,
    root_dirichlet_fraction: float = 0.25,
    root_dirichlet_alpha: float = 0.03,
    c_lcb: float = 0.0,
    gumbel_scale: float = 1.0,
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
        play_many_one = _build_play_many_games_fn_cpp(
            model=model,
            board_size=board_size,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            num_games=1,
            dynamic_considered_actions=dynamic_considered_actions,
            opening_considered_actions=opening_considered_actions,
            midgame_considered_actions=midgame_considered_actions,
            endgame_considered_actions=endgame_considered_actions,
            midgame_start_move=midgame_start_move,
            endgame_start_move=endgame_start_move,
            temperature_drop_move=temperature_drop_move,
            final_temperature=final_temperature,
            root_dirichlet_fraction=root_dirichlet_fraction,
            root_dirichlet_alpha=root_dirichlet_alpha,
            c_lcb=c_lcb,
            mode_freestyle_weight=1.0,
            mode_forbidden_weight=0.0,
            mode_swap2_weight=0.0,
            mode_swap2_forbidden_weight=0.0,
            swap2_swap_prob=0.5,
            gumbel_scale=gumbel_scale,
            mcts_backend=backend,
            cpp_virtual_loss=cpp_virtual_loss,
            cpp_c_puct=cpp_c_puct,
            cpp_num_threads=cpp_num_threads,
            cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
        )

        def play_one_fn(params, rng_key: jnp.ndarray, temperature: jnp.ndarray):
            obs, pi, value, mask, _action, steps, winners = play_many_one(params, rng_key, temperature)
            return (
                obs[0],
                pi[0],
                value[0],
                mask[0],
                steps[0],
                winners[0],
            )

        return play_one_fn

    max_steps = board_size * board_size
    num_actions = max_steps
    # Keep root-defense heuristics off for self-play throughput: they add
    # many gather/scatter/bincount ops to the MCTS jaxpr and can make TPU
    # compile + runtime dramatically slower for large batches.
    search_fn = build_search_fn(
        model=model,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
        root_dirichlet_fraction=root_dirichlet_fraction,
        root_dirichlet_alpha=root_dirichlet_alpha,
        force_defense_at_root=False,
        c_lcb=c_lcb,
        dynamic_considered_actions=dynamic_considered_actions,
        opening_considered_actions=opening_considered_actions,
        midgame_considered_actions=midgame_considered_actions,
        endgame_considered_actions=endgame_considered_actions,
        midgame_start_move=midgame_start_move,
        endgame_start_move=endgame_start_move,
        mcts_backend=backend,
        cpp_virtual_loss=cpp_virtual_loss,
        cpp_c_puct=cpp_c_puct,
        cpp_num_threads=cpp_num_threads,
        cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
    )

    @jax.jit
    def play_one_fn(params, rng_key: jnp.ndarray, temperature: jnp.ndarray):
        state = env.reset(board_size)
        obs_buf = jnp.zeros((max_steps, board_size, board_size, env.OBS_PLANES), dtype=jnp.float32)
        pi_buf = jnp.zeros((max_steps, num_actions), dtype=jnp.float32)
        to_play_buf = jnp.zeros((max_steps,), dtype=jnp.int8)

        def cond_fn(carry):
            cur_state, _, step_idx, _, _, _ = carry
            return (step_idx < max_steps) & (~cur_state.terminated)

        def body_fn(carry):
            cur_state, cur_key, step_idx, cur_obs, cur_pi, cur_to_play = carry
            cur_key, search_key, sample_key = jax.random.split(cur_key, 3)

            policy_output = search_fn(params, _add_batch_dim(cur_state), search_key)
            visit_probs = policy_output.action_weights[0]
            move_temperature = jnp.where(
                step_idx < jnp.int32(temperature_drop_move),
                temperature,
                jnp.float32(final_temperature),
            )
            action = _sample_action_jax(visit_probs, sample_key, move_temperature)

            obs = env.encode_state(cur_state)
            next_obs = cur_obs.at[step_idx].set(obs)
            next_pi = cur_pi.at[step_idx].set(visit_probs)
            next_to_play = cur_to_play.at[step_idx].set(cur_state.to_play)

            next_state, _, _ = env.step(cur_state, action)
            return (next_state, cur_key, step_idx + 1, next_obs, next_pi, next_to_play)

        state, _, steps, obs_buf, pi_buf, to_play_buf = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, rng_key, jnp.int32(0), obs_buf, pi_buf, to_play_buf),
        )

        winner = state.winner.astype(jnp.int8)
        mask = jnp.arange(max_steps, dtype=jnp.int32) < steps
        signed_values = jnp.where(
            winner == 0,
            jnp.float32(0.0),
            jnp.where(to_play_buf == winner, jnp.float32(1.0), jnp.float32(-1.0)),
        )
        value_buf = jnp.where(mask, signed_values, jnp.float32(0.0))
        pi_buf = jnp.where(mask[:, None], pi_buf, jnp.float32(0.0))
        return obs_buf, pi_buf, value_buf, mask, steps, winner

    return play_one_fn


def build_play_many_games_fn(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    num_games: int,
    dynamic_considered_actions: bool = False,
    opening_considered_actions: int = 64,
    midgame_considered_actions: int = 96,
    endgame_considered_actions: int = 160,
    midgame_start_move: int = 12,
    endgame_start_move: int = 40,
    temperature_drop_move: int = 12,
    final_temperature: float = 0.0,
    root_dirichlet_fraction: float = 0.25,
    root_dirichlet_alpha: float = 0.03,
    c_lcb: float = 0.0,
    mode_freestyle_weight: float = 1.0,
    mode_forbidden_weight: float = 0.0,
    mode_swap2_weight: float = 0.0,
    mode_swap2_forbidden_weight: float = 0.0,
    swap2_swap_prob: float = 0.5,
    gumbel_scale: float = 1.0,
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
        return _build_play_many_games_fn_cpp(
            model=model,
            board_size=board_size,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            num_games=num_games,
            dynamic_considered_actions=dynamic_considered_actions,
            opening_considered_actions=opening_considered_actions,
            midgame_considered_actions=midgame_considered_actions,
            endgame_considered_actions=endgame_considered_actions,
            midgame_start_move=midgame_start_move,
            endgame_start_move=endgame_start_move,
            temperature_drop_move=temperature_drop_move,
            final_temperature=final_temperature,
            root_dirichlet_fraction=root_dirichlet_fraction,
            root_dirichlet_alpha=root_dirichlet_alpha,
            c_lcb=c_lcb,
            mode_freestyle_weight=mode_freestyle_weight,
            mode_forbidden_weight=mode_forbidden_weight,
            mode_swap2_weight=mode_swap2_weight,
            mode_swap2_forbidden_weight=mode_swap2_forbidden_weight,
            swap2_swap_prob=swap2_swap_prob,
            gumbel_scale=gumbel_scale,
            mcts_backend=backend,
            cpp_virtual_loss=cpp_virtual_loss,
            cpp_c_puct=cpp_c_puct,
            cpp_num_threads=cpp_num_threads,
            cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
        )

    max_steps = board_size * board_size
    num_actions = max_steps
    # Keep root-defense heuristics off for self-play throughput: they add
    # many gather/scatter/bincount ops to the MCTS jaxpr and can make TPU
    # compile + runtime dramatically slower for large batches.
    search_fn = build_search_fn(
        model=model,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
        root_dirichlet_fraction=root_dirichlet_fraction,
        root_dirichlet_alpha=root_dirichlet_alpha,
        force_defense_at_root=False,
        c_lcb=c_lcb,
        dynamic_considered_actions=dynamic_considered_actions,
        opening_considered_actions=opening_considered_actions,
        midgame_considered_actions=midgame_considered_actions,
        endgame_considered_actions=endgame_considered_actions,
        midgame_start_move=midgame_start_move,
        endgame_start_move=endgame_start_move,
        mcts_backend=backend,
        cpp_virtual_loss=cpp_virtual_loss,
        cpp_c_puct=cpp_c_puct,
        cpp_num_threads=cpp_num_threads,
        cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
    )

    @jax.jit
    def play_many_fn(params, rng_key: jnp.ndarray, temperature: jnp.ndarray):
        setup_key, loop_key = jax.random.split(rng_key)
        batched_init_state = _build_mixed_init_state(
            board_size=board_size,
            num_games=num_games,
            rng_key=setup_key,
            mode_freestyle_weight=mode_freestyle_weight,
            mode_forbidden_weight=mode_forbidden_weight,
            mode_swap2_weight=mode_swap2_weight,
            mode_swap2_forbidden_weight=mode_swap2_forbidden_weight,
            swap2_swap_prob=swap2_swap_prob,
        )

        obs_buf = jnp.zeros((num_games, max_steps, board_size, board_size, env.OBS_PLANES), dtype=jnp.float32)
        pi_buf = jnp.zeros((num_games, max_steps, num_actions), dtype=jnp.float32)
        to_play_buf = jnp.zeros((num_games, max_steps), dtype=jnp.int8)
        mask_buf = jnp.zeros((num_games, max_steps), dtype=jnp.bool_)
        action_buf = jnp.full((num_games, max_steps), jnp.int32(-1), dtype=jnp.int32)
        steps = jnp.zeros((num_games,), dtype=jnp.int32)

        def cond_fn(carry):
            cur_state, _, step_idx, _, _, _, _, _, _ = carry
            return (step_idx < max_steps) & jnp.any(~cur_state.terminated)

        def body_fn(carry):
            cur_state, cur_key, step_idx, cur_obs, cur_pi, cur_to_play, cur_mask, cur_action, cur_steps = carry
            cur_key, search_key, sample_key = jax.random.split(cur_key, 3)

            active = ~cur_state.terminated

            def _replace_terminated(field, init_field):
                mask = active
                while mask.ndim < field.ndim:
                    mask = mask[..., None]
                return jnp.where(mask, field, init_field)

            search_state = jax.tree_util.tree_map(_replace_terminated, cur_state, batched_init_state)
            policy_output = search_fn(params, search_state, search_key)
            visit_probs = policy_output.action_weights
            move_temperature = jnp.where(
                step_idx < jnp.int32(temperature_drop_move),
                temperature,
                jnp.float32(final_temperature),
            )
            action = _sample_actions_jax(visit_probs, sample_key, move_temperature)

            obs = env.batch_encode_states(cur_state)
            obs_to_store = jnp.where(active[:, None, None, None], obs, jnp.float32(0.0))
            pi_to_store = jnp.where(active[:, None], visit_probs, jnp.float32(0.0))
            to_play_to_store = jnp.where(active, cur_state.to_play, jnp.int8(0))
            action_to_store = jnp.where(active, action, jnp.int32(-1))

            next_obs = cur_obs.at[:, step_idx].set(obs_to_store)
            next_pi = cur_pi.at[:, step_idx].set(pi_to_store)
            next_to_play = cur_to_play.at[:, step_idx].set(to_play_to_store)
            next_mask = cur_mask.at[:, step_idx].set(active)
            next_action = cur_action.at[:, step_idx].set(action_to_store)
            next_steps = cur_steps + active.astype(jnp.int32)

            next_state, _, _ = env.batch_step(cur_state, action)
            return (
                next_state,
                cur_key,
                step_idx + 1,
                next_obs,
                next_pi,
                next_to_play,
                next_mask,
                next_action,
                next_steps,
            )

        state, _, _, obs_buf, pi_buf, to_play_buf, mask_buf, action_buf, steps = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                batched_init_state,
                loop_key,
                jnp.int32(0),
                obs_buf,
                pi_buf,
                to_play_buf,
                mask_buf,
                action_buf,
                steps,
            ),
        )

        winners = state.winner.astype(jnp.int8)
        signed_values = jnp.where(
            winners[:, None] == 0,
            jnp.float32(0.0),
            jnp.where(to_play_buf == winners[:, None], jnp.float32(1.0), jnp.float32(-1.0)),
        )
        value_buf = jnp.where(mask_buf, signed_values, jnp.float32(0.0))
        pi_buf = jnp.where(mask_buf[:, :, None], pi_buf, jnp.float32(0.0))
        return obs_buf, pi_buf, value_buf, mask_buf, action_buf, steps, winners

    return play_many_fn


def play_one_game(
    *,
    params,
    model: PolicyValueNet,
    rng_key: jnp.ndarray,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    temperature: float,
    dynamic_considered_actions: bool = False,
    opening_considered_actions: int = 64,
    midgame_considered_actions: int = 96,
    endgame_considered_actions: int = 160,
    midgame_start_move: int = 12,
    endgame_start_move: int = 40,
    temperature_drop_move: int = 12,
    final_temperature: float = 0.0,
    root_dirichlet_fraction: float = 0.25,
    root_dirichlet_alpha: float = 0.03,
    c_lcb: float = 0.0,
    gumbel_scale: float = 1.0,
    mcts_backend: str = "mctx",
    cpp_virtual_loss: float = 1.0,
    cpp_c_puct: float = 1.5,
    cpp_num_threads: int = 0,
    cpp_leaf_eval_batch_size: int = 512,
) -> Tuple[List[TrainingExample], int]:
    compiled = build_play_one_game_fn(
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        dynamic_considered_actions=dynamic_considered_actions,
        opening_considered_actions=opening_considered_actions,
        midgame_considered_actions=midgame_considered_actions,
        endgame_considered_actions=endgame_considered_actions,
        midgame_start_move=midgame_start_move,
        endgame_start_move=endgame_start_move,
        temperature_drop_move=temperature_drop_move,
        final_temperature=final_temperature,
        root_dirichlet_fraction=root_dirichlet_fraction,
        root_dirichlet_alpha=root_dirichlet_alpha,
        c_lcb=c_lcb,
        gumbel_scale=gumbel_scale,
        mcts_backend=mcts_backend,
        cpp_virtual_loss=cpp_virtual_loss,
        cpp_c_puct=cpp_c_puct,
        cpp_num_threads=cpp_num_threads,
        cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
    )
    obs_buf, pi_buf, value_buf, mask, _, winner = compiled(params, rng_key, jnp.float32(temperature))

    obs_np = jax.device_get(obs_buf)
    pi_np = jax.device_get(pi_buf)
    value_np = jax.device_get(value_buf)
    mask_np = jax.device_get(mask)

    valid_idx = np.flatnonzero(mask_np)
    examples: List[TrainingExample] = [
        TrainingExample(
            observation=obs_np[idx],
            policy_target=pi_np[idx],
            value_target=float(value_np[idx]),
        )
        for idx in valid_idx
    ]
    return examples, int(winner)


def play_many_games(
    *,
    params,
    model: PolicyValueNet,
    rng_key: jnp.ndarray,
    num_games: int,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    temperature: float,
    dynamic_considered_actions: bool = False,
    opening_considered_actions: int = 64,
    midgame_considered_actions: int = 96,
    endgame_considered_actions: int = 160,
    midgame_start_move: int = 12,
    endgame_start_move: int = 40,
    temperature_drop_move: int = 12,
    final_temperature: float = 0.0,
    root_dirichlet_fraction: float = 0.25,
    root_dirichlet_alpha: float = 0.03,
    c_lcb: float = 0.0,
    mode_freestyle_weight: float = 1.0,
    mode_forbidden_weight: float = 0.0,
    mode_swap2_weight: float = 0.0,
    mode_swap2_forbidden_weight: float = 0.0,
    swap2_swap_prob: float = 0.5,
    gumbel_scale: float = 1.0,
    mcts_backend: str = "mctx",
    cpp_virtual_loss: float = 1.0,
    cpp_c_puct: float = 1.5,
    cpp_num_threads: int = 0,
    cpp_leaf_eval_batch_size: int = 512,
) -> Tuple[List[TrainingExample], dict]:
    play_many_fn = build_play_many_games_fn(
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        num_games=num_games,
        dynamic_considered_actions=dynamic_considered_actions,
        opening_considered_actions=opening_considered_actions,
        midgame_considered_actions=midgame_considered_actions,
        endgame_considered_actions=endgame_considered_actions,
        midgame_start_move=midgame_start_move,
        endgame_start_move=endgame_start_move,
        temperature_drop_move=temperature_drop_move,
        final_temperature=final_temperature,
        root_dirichlet_fraction=root_dirichlet_fraction,
        root_dirichlet_alpha=root_dirichlet_alpha,
        c_lcb=c_lcb,
        mode_freestyle_weight=mode_freestyle_weight,
        mode_forbidden_weight=mode_forbidden_weight,
        mode_swap2_weight=mode_swap2_weight,
        mode_swap2_forbidden_weight=mode_swap2_forbidden_weight,
        swap2_swap_prob=swap2_swap_prob,
        gumbel_scale=gumbel_scale,
        mcts_backend=mcts_backend,
        cpp_virtual_loss=cpp_virtual_loss,
        cpp_c_puct=cpp_c_puct,
        cpp_num_threads=cpp_num_threads,
        cpp_leaf_eval_batch_size=cpp_leaf_eval_batch_size,
    )
    obs, pi, value, mask, _, _, winners = play_many_fn(params, rng_key, jnp.float32(temperature))
    obs = jax.device_get(obs)
    pi = jax.device_get(pi)
    value = jax.device_get(value)
    mask = jax.device_get(mask)
    winners = jax.device_get(winners)

    flat_obs = obs.reshape((-1, board_size, board_size, env.OBS_PLANES))
    flat_pi = pi.reshape((-1, board_size * board_size))
    flat_value = value.reshape((-1,))
    flat_mask = mask.reshape((-1,))
    valid_idx = np.flatnonzero(flat_mask)
    all_examples: List[TrainingExample] = [
        TrainingExample(
            observation=flat_obs[idx],
            policy_target=flat_pi[idx],
            value_target=float(flat_value[idx]),
        )
        for idx in valid_idx
    ]

    stats = {"black_win": int((winners == 1).sum()), "white_win": int((winners == -1).sum()), "draw": int((winners == 0).sum())}
    return all_examples, stats


def stack_examples(examples: Sequence[TrainingExample]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs = jnp.stack([x.observation for x in examples], axis=0)
    policy = jnp.stack([x.policy_target for x in examples], axis=0)
    value = jnp.asarray([x.value_target for x in examples], dtype=jnp.float32)
    return obs, policy, value


def main() -> None:
    configure_jax_runtime(app_name="self_play", repo_root=REPO_ROOT)
    parser = argparse.ArgumentParser(description="Run minimal JAX+mctx gomoku self-play.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--max-num-considered-actions", type=int, default=24)
    parser.add_argument("--disable-dynamic-considered-actions", action="store_true")
    parser.add_argument("--considered-actions-opening", type=int, default=64)
    parser.add_argument("--considered-actions-midgame", type=int, default=96)
    parser.add_argument("--considered-actions-endgame", type=int, default=160)
    parser.add_argument("--considered-actions-mid-move", type=int, default=12)
    parser.add_argument("--considered-actions-endgame-move", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--c-lcb", type=float, default=0.0)
    parser.add_argument("--gumbel-scale", type=float, default=1.0)
    parser.add_argument("--mcts-backend", choices=("mctx", "cpp"), default="mctx")
    parser.add_argument("--cpp-virtual-loss", type=float, default=1.0)
    parser.add_argument("--cpp-c-puct", type=float, default=1.5)
    parser.add_argument("--cpp-num-threads", type=int, default=0)
    parser.add_argument("--cpp-leaf-eval-batch-size", type=int, default=512)
    parser.add_argument("--max-attention-heads", type=int, default=4)
    parser.add_argument("--compute-dtype", type=str, default="bfloat16")
    parser.add_argument("--param-dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    compute_dtype = _dtype_from_name(args.compute_dtype)
    param_dtype = _dtype_from_name(args.param_dtype)
    model = PolicyValueNet(
        board_size=args.board_size,
        max_attention_heads=args.max_attention_heads,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    init_key, game_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = model.init(init_key, jnp.zeros((1, args.board_size, args.board_size, env.OBS_PLANES), dtype=compute_dtype))
    samples, winner = play_one_game(
        params=params,
        model=model,
        rng_key=game_key,
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        max_num_considered_actions=args.max_num_considered_actions,
        temperature=args.temperature,
        dynamic_considered_actions=not args.disable_dynamic_considered_actions,
        opening_considered_actions=args.considered_actions_opening,
        midgame_considered_actions=args.considered_actions_midgame,
        endgame_considered_actions=args.considered_actions_endgame,
        midgame_start_move=args.considered_actions_mid_move,
        endgame_start_move=args.considered_actions_endgame_move,
        c_lcb=args.c_lcb,
        gumbel_scale=args.gumbel_scale,
        mcts_backend=args.mcts_backend,
        cpp_virtual_loss=args.cpp_virtual_loss,
        cpp_c_puct=args.cpp_c_puct,
        cpp_num_threads=args.cpp_num_threads,
        cpp_leaf_eval_batch_size=args.cpp_leaf_eval_batch_size,
    )
    print(f"samples={len(samples)} winner={winner}")


if __name__ == "__main__":
    main()
