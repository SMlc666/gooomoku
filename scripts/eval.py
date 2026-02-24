from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku import env
from gooomoku.mctx_adapter import run_gumbel_search
from gooomoku.net import PolicyValueNet


def _add_batch_dim(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _load_checkpoint(path: Path):
    with path.open("rb") as fp:
        payload = pickle.load(fp)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    config = payload.get("config", {})
    return params, config


def _random_legal_action(state: env.GomokuState, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(np.asarray(env.legal_action_mask(state)))
    if legal.size == 0:
        return 0
    return int(rng.choice(legal))


def _agent_action(
    *,
    state: env.GomokuState,
    params,
    model: PolicyValueNet,
    rng_key,
    num_simulations: int,
    max_num_considered_actions: int,
) -> int:
    policy_output = run_gumbel_search(
        params=params,
        model=model,
        states=_add_batch_dim(state),
        rng_key=rng_key,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )
    return int(jnp.argmax(policy_output.action_weights[0]))


def play_vs_random(
    *,
    params,
    model: PolicyValueNet,
    seed: int,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    agent_color: int,
) -> int:
    state = env.reset(board_size)
    np_rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    for _ in range(board_size * board_size):
        if bool(state.terminated):
            break

        if int(state.to_play) == agent_color:
            jax_key, key = jax.random.split(jax_key)
            action = _agent_action(
                state=state,
                params=params,
                model=model,
                rng_key=key,
                num_simulations=num_simulations,
                max_num_considered_actions=max_num_considered_actions,
            )
        else:
            action = _random_legal_action(state, np_rng)

        state, _, _ = env.step(state, jnp.int32(action))
    return int(state.winner)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against random agent.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pkl"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-num-considered-actions", type=int, default=None)
    args = parser.parse_args()

    params, config = _load_checkpoint(args.checkpoint)
    board_size = args.board_size or int(config.get("board_size", 9))
    channels = args.channels or int(config.get("channels", 64))
    blocks = args.blocks or int(config.get("blocks", 6))
    num_simulations = args.num_simulations or int(config.get("num_simulations", 64))
    max_num_considered_actions = args.max_num_considered_actions or int(config.get("max_num_considered_actions", 16))

    model = PolicyValueNet(board_size=board_size, channels=channels, blocks=blocks)

    win = 0
    loss = 0
    draw = 0
    for game_idx in range(args.games):
        agent_color = 1 if (game_idx % 2 == 0) else -1
        winner = play_vs_random(
            params=params,
            model=model,
            seed=args.seed + game_idx,
            board_size=board_size,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            agent_color=agent_color,
        )
        if winner == 0:
            draw += 1
        elif winner == agent_color:
            win += 1
        else:
            loss += 1

    print(
        f"games={args.games} win={win} loss={loss} draw={draw} "
        f"win_rate={win / max(args.games, 1):.3f}"
    )


if __name__ == "__main__":
    main()
