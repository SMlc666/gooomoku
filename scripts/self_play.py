from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku import env
from gooomoku.mctx_adapter import run_gumbel_search
from gooomoku.net import PolicyValueNet


@dataclass
class TrainingExample:
    observation: jnp.ndarray
    policy_target: jnp.ndarray
    value_target: float


def _add_batch_dim(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _sample_action(visit_probs: jnp.ndarray, rng_key: jnp.ndarray, temperature: float) -> int:
    if temperature <= 1e-6:
        return int(jnp.argmax(visit_probs))
    logits = jnp.log(visit_probs + 1e-8) / jnp.float32(temperature)
    return int(jax.random.categorical(rng_key, logits))


def play_one_game(
    *,
    params,
    model: PolicyValueNet,
    rng_key: jnp.ndarray,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    temperature: float,
) -> Tuple[List[TrainingExample], int]:
    state = env.reset(board_size)
    trajectory: List[Tuple[jnp.ndarray, jnp.ndarray, int]] = []

    for _ in range(board_size * board_size):
        if bool(state.terminated):
            break

        rng_key, search_key, sample_key = jax.random.split(rng_key, 3)
        policy_output = run_gumbel_search(
            params=params,
            model=model,
            states=_add_batch_dim(state),
            rng_key=search_key,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
        )

        visit_probs = policy_output.action_weights[0]
        action = _sample_action(visit_probs, sample_key, temperature=temperature)
        trajectory.append((env.encode_state(state), visit_probs, int(state.to_play)))
        state, _, _ = env.step(state, jnp.int32(action))

    winner = int(state.winner)
    examples: List[TrainingExample] = []
    for obs, pi, to_play in trajectory:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == to_play else -1.0
        examples.append(TrainingExample(observation=obs, policy_target=pi, value_target=value))
    return examples, winner


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
) -> Tuple[List[TrainingExample], dict]:
    all_examples: List[TrainingExample] = []
    stats = {"black_win": 0, "white_win": 0, "draw": 0}
    for _ in range(num_games):
        rng_key, game_key = jax.random.split(rng_key)
        examples, winner = play_one_game(
            params=params,
            model=model,
            rng_key=game_key,
            board_size=board_size,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            temperature=temperature,
        )
        all_examples.extend(examples)
        if winner == 1:
            stats["black_win"] += 1
        elif winner == -1:
            stats["white_win"] += 1
        else:
            stats["draw"] += 1
    return all_examples, stats


def stack_examples(examples: Sequence[TrainingExample]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs = jnp.stack([x.observation for x in examples], axis=0)
    policy = jnp.stack([x.policy_target for x in examples], axis=0)
    value = jnp.asarray([x.value_target for x in examples], dtype=jnp.float32)
    return obs, policy, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal JAX+mctx gomoku self-play.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--max-num-considered-actions", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PolicyValueNet(board_size=args.board_size)
    init_key, game_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = model.init(init_key, jnp.zeros((1, args.board_size, args.board_size, 4), dtype=jnp.float32))
    samples, winner = play_one_game(
        params=params,
        model=model,
        rng_key=game_key,
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        max_num_considered_actions=args.max_num_considered_actions,
        temperature=args.temperature,
    )
    print(f"samples={len(samples)} winner={winner}")


if __name__ == "__main__":
    main()
