from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku import env
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


def _add_batch_dim(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _load_checkpoint(path: Path):
    with path.open("rb") as fp:
        payload = pickle.load(fp)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    config = payload.get("config", {})
    return params, config


def build_play_vs_random_fn(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
) -> callable:
    max_steps = board_size * board_size
    search_fn = build_search_fn(
        model=model,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        root_dirichlet_fraction=0.0,
        root_dirichlet_alpha=0.03,
    )

    @jax.jit
    def play_vs_random_fn(params, rng_key: jnp.ndarray, agent_color: jnp.ndarray):
        state = env.reset(board_size)

        def cond_fn(carry):
            cur_state, _, step_idx = carry
            return (step_idx < max_steps) & (~cur_state.terminated)

        def body_fn(carry):
            cur_state, cur_key, step_idx = carry
            cur_key, agent_key, random_key = jax.random.split(cur_key, 3)

            def agent_action(_):
                policy_output = search_fn(params, _add_batch_dim(cur_state), agent_key)
                return jnp.argmax(policy_output.action_weights[0]).astype(jnp.int32)

            def random_action(_):
                legal = env.legal_action_mask(cur_state)
                logits = jnp.where(legal, jnp.float32(0.0), jnp.float32(-1e9))
                return jax.random.categorical(random_key, logits).astype(jnp.int32)

            action = jax.lax.cond(
                cur_state.to_play == agent_color,
                agent_action,
                random_action,
                operand=None,
            )
            next_state, _, _ = env.step(cur_state, action)
            return (next_state, cur_key, step_idx + 1)

        state, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, rng_key, jnp.int32(0)),
        )
        return state.winner.astype(jnp.int8)

    return play_vs_random_fn


def build_eval_vs_random_fn(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    num_games: int,
) -> callable:
    play_vs_random_fn = build_play_vs_random_fn(
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
    )

    @jax.jit
    def eval_fn(params, rng_key: jnp.ndarray):
        game_indices = jnp.arange(num_games, dtype=jnp.int32)

        def body_fn(carry, game_idx):
            cur_key, win, loss, draw = carry
            cur_key, game_key = jax.random.split(cur_key)
            agent_color = jnp.where((game_idx % 2) == 0, jnp.int8(1), jnp.int8(-1))
            winner = play_vs_random_fn(params, game_key, agent_color)
            is_draw = winner == 0
            is_win = winner == agent_color
            win = win + is_win.astype(jnp.int32)
            draw = draw + is_draw.astype(jnp.int32)
            loss = loss + ((~is_draw) & (~is_win)).astype(jnp.int32)
            return (cur_key, win, loss, draw), winner

        (_, win, loss, draw), winners = jax.lax.scan(
            body_fn,
            (rng_key, jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            game_indices,
        )
        return win, loss, draw, winners

    return eval_fn


def main() -> None:
    configure_jax_runtime(app_name="eval", repo_root=REPO_ROOT)
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against random agent.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pkl"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--compute-dtype", type=str, default=None)
    parser.add_argument("--param-dtype", type=str, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-num-considered-actions", type=int, default=None)
    args = parser.parse_args()

    params, config = _load_checkpoint(args.checkpoint)
    board_size = args.board_size or int(config.get("board_size", 9))
    channels = args.channels or int(config.get("channels", 64))
    blocks = args.blocks or int(config.get("blocks", 6))
    compute_dtype_name = args.compute_dtype or str(config.get("compute_dtype", "float32"))
    param_dtype_name = args.param_dtype or str(config.get("param_dtype", "float32"))
    compute_dtype = _dtype_from_name(compute_dtype_name)
    param_dtype = _dtype_from_name(param_dtype_name)
    num_simulations = args.num_simulations or int(config.get("num_simulations", 64))
    max_num_considered_actions = args.max_num_considered_actions or int(config.get("max_num_considered_actions", 24))

    model = PolicyValueNet(
        board_size=board_size,
        channels=channels,
        blocks=blocks,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )

    eval_fn = build_eval_vs_random_fn(
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        num_games=args.games,
    )
    win, loss, draw, _ = eval_fn(params, jax.random.PRNGKey(args.seed))
    win = int(win)
    loss = int(loss)
    draw = int(draw)

    print(
        f"games={args.games} win={win} loss={loss} draw={draw} "
        f"win_rate={win / max(args.games, 1):.3f}"
    )


if __name__ == "__main__":
    main()
