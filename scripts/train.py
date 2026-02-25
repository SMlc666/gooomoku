from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku import env
from gooomoku.mctx_adapter import build_search_fn
from gooomoku.net import PolicyValueNet
from scripts.self_play import build_play_many_games_fn


def _dtype_from_name(name: str):
    table = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[name]


def _l2_regularization(params) -> jnp.ndarray:
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        params,
        initializer=jnp.float32(0.0),
    )


def make_single_train_step(model: PolicyValueNet, optimizer: optax.GradientTransformation, weight_decay: float):
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, obs, policy_target, value_target):
        def loss_fn(trainable):
            logits, value = model.apply(trainable, obs.astype(model.compute_dtype))
            logits = logits.astype(jnp.float32)
            value = value.astype(jnp.float32)
            policy_loss = -jnp.mean(jnp.sum(policy_target * jax.nn.log_softmax(logits), axis=-1))
            value_loss = jnp.mean(jnp.square(value - value_target))
            reg = jnp.float32(weight_decay) * _l2_regularization(trainable)
            total_loss = policy_loss + value_loss + reg
            return total_loss, (policy_loss, value_loss)

        (loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, policy_loss, value_loss

    return train_step


def make_pmap_train_step(model: PolicyValueNet, optimizer: optax.GradientTransformation, weight_decay: float):
    @functools.partial(jax.pmap, axis_name="device", donate_argnums=(0, 1))
    def train_step(params, opt_state, obs, policy_target, value_target):
        def loss_fn(trainable):
            logits, value = model.apply(trainable, obs.astype(model.compute_dtype))
            logits = logits.astype(jnp.float32)
            value = value.astype(jnp.float32)
            policy_loss = -jnp.mean(jnp.sum(policy_target * jax.nn.log_softmax(logits), axis=-1))
            value_loss = jnp.mean(jnp.square(value - value_target))
            reg = jnp.float32(weight_decay) * _l2_regularization(trainable)
            total_loss = policy_loss + value_loss + reg
            return total_loss, (policy_loss, value_loss)

        (loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.lax.pmean(grads, axis_name="device")
        loss = jax.lax.pmean(loss, axis_name="device")
        policy_loss = jax.lax.pmean(policy_loss, axis_name="device")
        value_loss = jax.lax.pmean(value_loss, axis_name="device")
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, policy_loss, value_loss

    return train_step


def make_pmap_collect_step(play_many_games_fn):
    @functools.partial(jax.pmap, axis_name="device")
    def collect_step(params, rng_key, temperature):
        return play_many_games_fn(params, rng_key, temperature)

    return collect_step


def make_pmap_arena_step(arena_fn):
    @functools.partial(jax.pmap, axis_name="device")
    def arena_step(params_a, params_b, rng_key):
        return arena_fn(params_a, params_b, rng_key)

    return arena_step


def _save_model_checkpoint(path: Path, params, config: dict) -> None:
    payload = {"params": jax.device_get(params), "config": config}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(payload, fp)


def _save_training_checkpoint(
    path: Path,
    *,
    params,
    opt_state,
    config: dict,
    step: int,
    optimizer_updates: int,
    rng_key,
    np_rng_state,
    replay_obs: np.ndarray,
    replay_policy: np.ndarray,
    replay_value: np.ndarray,
    best_params,
    best_step: int,
) -> None:
    payload = {
        "format_version": 2,
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "config": config,
        "step": int(step),
        "optimizer_updates": int(optimizer_updates),
        "rng_key": jax.device_get(rng_key),
        "np_rng_state": np_rng_state,
        "replay_obs": replay_obs,
        "replay_policy": replay_policy,
        "replay_value": replay_value,
        "best_params": jax.device_get(best_params),
        "best_step": int(best_step),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(payload, fp)


def _load_checkpoint_payload(path: Path):
    with path.open("rb") as fp:
        return pickle.load(fp)


def _extract_host_tree(params, use_pmap: bool):
    return jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params


def _add_batch_dim_state(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _checkpoint_config(args, optimizer_updates: int, best_step: int) -> dict:
    return {
        "board_size": args.board_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "num_simulations": args.num_simulations,
        "max_num_considered_actions": args.max_num_considered_actions,
        "updates_per_step": args.updates_per_step,
        "temperature": args.temperature,
        "temperature_drop_move": args.temperature_drop_move,
        "final_temperature": args.final_temperature,
        "root_dirichlet_fraction": args.root_dirichlet_fraction,
        "root_dirichlet_alpha": args.root_dirichlet_alpha,
        "lr": args.lr,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_end_value": args.lr_end_value,
        "compute_dtype": args.compute_dtype,
        "param_dtype": args.param_dtype,
        "symmetry_augmentation": not args.disable_symmetry_augmentation,
        "arena_every_steps": args.arena_every_steps,
        "arena_games": args.arena_games,
        "arena_replace_threshold": args.arena_replace_threshold,
        "arena_num_simulations": args.arena_num_simulations,
        "arena_max_num_considered_actions": args.arena_max_num_considered_actions,
        "arena_temperature": args.arena_temperature,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "optimizer_updates": optimizer_updates,
        "best_step": best_step,
    }


def build_arena_fn(
    *,
    model: PolicyValueNet,
    board_size: int,
    num_simulations: int,
    max_num_considered_actions: int,
    num_games: int,
    temperature: float,
):
    max_steps = board_size * board_size
    search_fn = build_search_fn(
        model=model,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        root_dirichlet_fraction=0.0,
        root_dirichlet_alpha=0.03,
    )
    use_greedy = temperature <= 1e-6
    temp = jnp.float32(temperature)

    @jax.jit
    def play_game(params_a, params_b, rng_key, a_color):
        state = env.reset(board_size)

        def cond_fn(carry):
            cur_state, _, step_idx = carry
            return (step_idx < max_steps) & (~cur_state.terminated)

        def body_fn(carry):
            cur_state, cur_key, step_idx = carry
            cur_key, search_key, sample_key = jax.random.split(cur_key, 3)

            def choose_action(eval_params):
                policy_output = search_fn(eval_params, _add_batch_dim_state(cur_state), search_key)
                visit_probs = policy_output.action_weights[0]
                if use_greedy:
                    return jnp.argmax(visit_probs).astype(jnp.int32)
                logits = jnp.log(jnp.maximum(visit_probs, 1e-8)) / temp
                return jax.random.categorical(sample_key, logits).astype(jnp.int32)

            action = jax.lax.cond(
                cur_state.to_play == a_color,
                lambda pair: choose_action(pair[0]),
                lambda pair: choose_action(pair[1]),
                operand=(params_a, params_b),
            )
            next_state, _, _ = env.step(cur_state, action)
            return (next_state, cur_key, step_idx + 1)

        state, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, rng_key, jnp.int32(0)),
        )
        winner = state.winner
        a_win = winner == a_color
        draw = winner == 0
        b_win = (~a_win) & (~draw)
        return a_win.astype(jnp.int32), b_win.astype(jnp.int32), draw.astype(jnp.int32)

    @jax.jit
    def arena_fn(params_a, params_b, rng_key):
        game_indices = jnp.arange(num_games, dtype=jnp.int32)

        def body_fn(carry, game_idx):
            cur_key, a_wins, b_wins, draws = carry
            cur_key, game_key = jax.random.split(cur_key)
            a_color = jnp.where((game_idx % 2) == 0, jnp.int8(1), jnp.int8(-1))
            a_win, b_win, draw = play_game(params_a, params_b, game_key, a_color)
            return (cur_key, a_wins + a_win, b_wins + b_win, draws + draw), 0

        (_, a_wins, b_wins, draws), _ = jax.lax.scan(
            body_fn,
            (rng_key, jnp.int32(0), jnp.int32(0), jnp.int32(0)),
            game_indices,
        )
        return a_wins, b_wins, draws

    return arena_fn


def _replicate_tree_for_pmap(tree, local_devices: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (local_devices,) + x.shape),
        tree,
    )


def _apply_dihedral_symmetry_obs(obs: np.ndarray, symmetry_id: int) -> np.ndarray:
    transformed = obs
    if symmetry_id >= 4:
        transformed = np.flip(transformed, axis=2)
        symmetry_id -= 4
    if symmetry_id > 0:
        transformed = np.rot90(transformed, k=symmetry_id, axes=(1, 2))
    return transformed


def _apply_dihedral_symmetry_policy(policy_grid: np.ndarray, symmetry_id: int) -> np.ndarray:
    transformed = policy_grid
    if symmetry_id >= 4:
        transformed = np.flip(transformed, axis=2)
        symmetry_id -= 4
    if symmetry_id > 0:
        transformed = np.rot90(transformed, k=symmetry_id, axes=(1, 2))
    return transformed


def _augment_batch_with_random_symmetry(
    obs: np.ndarray,
    policy: np.ndarray,
    board_size: int,
    np_rng: np.random.Generator,
):
    batch_size = obs.shape[0]
    symmetry_ids = np_rng.integers(0, 8, size=batch_size, dtype=np.int32)

    augmented_obs = np.empty_like(obs)
    policy_grid = policy.reshape((-1, board_size, board_size))
    augmented_policy_grid = np.empty_like(policy_grid)

    for symmetry_id in range(8):
        indices = np.nonzero(symmetry_ids == symmetry_id)[0]
        if indices.size == 0:
            continue
        augmented_obs[indices] = _apply_dihedral_symmetry_obs(obs[indices], symmetry_id)
        augmented_policy_grid[indices] = _apply_dihedral_symmetry_policy(policy_grid[indices], symmetry_id)

    return augmented_obs, augmented_policy_grid.reshape((-1, board_size * board_size))


def _append_replay(
    replay_obs: np.ndarray,
    replay_policy: np.ndarray,
    replay_value: np.ndarray,
    new_obs: np.ndarray,
    new_policy: np.ndarray,
    new_value: np.ndarray,
    replay_size: int,
):
    if replay_obs.size == 0:
        merged_obs = new_obs
        merged_policy = new_policy
        merged_value = new_value
    else:
        merged_obs = np.concatenate([replay_obs, new_obs], axis=0)
        merged_policy = np.concatenate([replay_policy, new_policy], axis=0)
        merged_value = np.concatenate([replay_value, new_value], axis=0)

    if merged_obs.shape[0] > replay_size:
        start = merged_obs.shape[0] - replay_size
        merged_obs = merged_obs[start:]
        merged_policy = merged_policy[start:]
        merged_value = merged_value[start:]
    return merged_obs, merged_policy, merged_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal JAX+mctx gomoku trainer.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--games-per-step", type=int, default=8)
    parser.add_argument("--updates-per-step", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--lr-end-value", type=float, default=1e-4)
    parser.add_argument("--compute-dtype", type=str, default="bfloat16")
    parser.add_argument("--param-dtype", type=str, default="float32")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--max-num-considered-actions", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=12)
    parser.add_argument("--final-temperature", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--disable-symmetry-augmentation", action="store_true")
    parser.add_argument("--arena-every-steps", type=int, default=10)
    parser.add_argument("--arena-games", type=int, default=64)
    parser.add_argument("--arena-replace-threshold", type=float, default=0.55)
    parser.add_argument("--arena-num-simulations", type=int, default=None)
    parser.add_argument("--arena-max-num-considered-actions", type=int, default=None)
    parser.add_argument("--arena-temperature", type=float, default=0.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest.pkl"))
    parser.add_argument("--disable-pmap", action="store_true")
    args = parser.parse_args()

    if args.updates_per_step < 1:
        raise ValueError("updates-per-step must be >= 1")
    compute_dtype = _dtype_from_name(args.compute_dtype)
    param_dtype = _dtype_from_name(args.param_dtype)
    if args.arena_every_steps < 0:
        raise ValueError("arena-every-steps must be >= 0")
    if args.arena_games < 2:
        raise ValueError("arena-games must be >= 2")
    if args.arena_games % 2 != 0:
        raise ValueError("arena-games must be even")
    if not (0.0 < args.arena_replace_threshold <= 1.0):
        raise ValueError("arena-replace-threshold must be in (0, 1]")
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint-every-steps must be >= 0")

    model = PolicyValueNet(
        board_size=args.board_size,
        channels=args.channels,
        blocks=args.blocks,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )
    rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = model.init(init_key, jnp.zeros((1, args.board_size, args.board_size, 4), dtype=compute_dtype))

    total_optimizer_updates = max(1, args.train_steps * args.updates_per_step)
    warmup_steps = max(0, min(args.lr_warmup_steps, total_optimizer_updates - 1))
    decay_steps = max(total_optimizer_updates, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=args.lr_end_value,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)
    replay_obs = np.zeros((0, args.board_size, args.board_size, 4), dtype=np.float32)
    replay_policy = np.zeros((0, args.board_size * args.board_size), dtype=np.float32)
    replay_value = np.zeros((0,), dtype=np.float32)
    np_rng = np.random.default_rng(args.seed + 13)
    optimizer_updates = 0
    best_params = params
    best_step = 0
    start_step = 1

    if args.resume_from is not None:
        payload = _load_checkpoint_payload(args.resume_from)
        resume_cfg = payload.get("config", {})
        for key in ("board_size", "channels", "blocks", "compute_dtype", "param_dtype"):
            cfg_val = resume_cfg.get(key)
            arg_val = getattr(args, key)
            if cfg_val is not None and str(cfg_val) != str(arg_val):
                raise ValueError(
                    f"resume config mismatch for {key}: checkpoint={cfg_val} current={arg_val}"
                )

        if "params" not in payload:
            raise ValueError(f"invalid resume checkpoint (missing params): {args.resume_from}")
        params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
        if "opt_state" in payload:
            opt_state = jax.tree_util.tree_map(jnp.asarray, payload["opt_state"])
        else:
            opt_state = optimizer.init(params)

        if "rng_key" in payload:
            rng_key = jnp.asarray(payload["rng_key"], dtype=jnp.uint32)

        if "replay_obs" in payload and "replay_policy" in payload and "replay_value" in payload:
            replay_obs = np.asarray(payload["replay_obs"], dtype=np.float32)
            replay_policy = np.asarray(payload["replay_policy"], dtype=np.float32)
            replay_value = np.asarray(payload["replay_value"], dtype=np.float32)

        np_rng = np.random.default_rng(args.seed + 13)
        if "np_rng_state" in payload:
            np_rng.bit_generator.state = payload["np_rng_state"]

        optimizer_updates = int(payload.get("optimizer_updates", 0))
        last_step = int(payload.get("step", 0))
        start_step = max(1, last_step + 1)
        if "best_params" in payload:
            best_params = jax.tree_util.tree_map(jnp.asarray, payload["best_params"])
        else:
            best_params = params
        best_step = int(payload.get("best_step", 0))

        print(
            f"resumed from {args.resume_from}: last_step={last_step} start_step={start_step} "
            f"optimizer_updates={optimizer_updates} replay={replay_obs.shape[0]}"
        )

    local_devices = jax.local_device_count()
    use_pmap = (not args.disable_pmap) and local_devices > 1
    if use_pmap and (args.batch_size % local_devices != 0):
        raise ValueError(f"batch-size must be divisible by local_device_count={local_devices}")
    if use_pmap and (args.games_per_step % local_devices != 0):
        raise ValueError(f"games-per-step must be divisible by local_device_count={local_devices}")
    if use_pmap and args.arena_every_steps > 0 and (args.arena_games % local_devices != 0):
        raise ValueError(f"arena-games must be divisible by local_device_count={local_devices} when pmap is enabled")

    games_per_device = args.games_per_step // local_devices if use_pmap else args.games_per_step
    play_many_games_fn = build_play_many_games_fn(
        model=model,
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        max_num_considered_actions=args.max_num_considered_actions,
        num_games=games_per_device,
        temperature_drop_move=args.temperature_drop_move,
        final_temperature=args.final_temperature,
        root_dirichlet_fraction=args.root_dirichlet_fraction,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
    )
    arena_num_simulations = args.arena_num_simulations or args.num_simulations
    arena_max_num_considered_actions = args.arena_max_num_considered_actions or args.max_num_considered_actions
    arena_games_per_device = args.arena_games // local_devices if use_pmap else args.arena_games
    arena_fn = build_arena_fn(
        model=model,
        board_size=args.board_size,
        num_simulations=arena_num_simulations,
        max_num_considered_actions=arena_max_num_considered_actions,
        num_games=arena_games_per_device,
        temperature=args.arena_temperature,
    )
    pmap_arena_step = make_pmap_arena_step(arena_fn) if use_pmap else None

    if use_pmap:
        per_device_batch = args.batch_size // local_devices
        train_step = make_pmap_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = make_pmap_collect_step(play_many_games_fn)
        params = _replicate_tree_for_pmap(params, local_devices=local_devices)
        opt_state = _replicate_tree_for_pmap(opt_state, local_devices=local_devices)
        best_params_repl = _replicate_tree_for_pmap(best_params, local_devices=local_devices)
        print(
            f"pmap enabled: local_device_count={local_devices}, per_device_batch={per_device_batch}, "
            f"selfplay_games_per_device={games_per_device}, updates_per_step={args.updates_per_step}, "
            f"compute_dtype={args.compute_dtype}, param_dtype={args.param_dtype}"
        )
    else:
        train_step = make_single_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = play_many_games_fn
        per_device_batch = args.batch_size
        best_params_repl = None
        print(
            f"single-device mode: local_device_count={local_devices}, updates_per_step={args.updates_per_step}, "
            f"compute_dtype={args.compute_dtype}, param_dtype={args.param_dtype}"
        )

    best_params = jax.tree_util.tree_map(jnp.asarray, best_params)
    best_path = args.output.parent / f"{args.output.stem}.best{args.output.suffix}"

    def maybe_save_training_checkpoint(step: int, reason: str) -> None:
        if args.checkpoint_every_steps <= 0 or (step % args.checkpoint_every_steps != 0):
            return
        current_params = _extract_host_tree(params, use_pmap=use_pmap)
        current_opt_state = _extract_host_tree(opt_state, use_pmap=use_pmap)
        cfg = _checkpoint_config(args, optimizer_updates=optimizer_updates, best_step=best_step)
        _save_training_checkpoint(
            args.output,
            params=current_params,
            opt_state=current_opt_state,
            config=cfg,
            step=step,
            optimizer_updates=optimizer_updates,
            rng_key=rng_key,
            np_rng_state=np_rng.bit_generator.state,
            replay_obs=replay_obs,
            replay_policy=replay_policy,
            replay_value=replay_value,
            best_params=best_params,
            best_step=best_step,
        )
        print(f"saved training checkpoint ({reason}) to {args.output}")

    completed_step = start_step - 1
    if start_step > args.train_steps:
        print(
            f"resume start_step={start_step} is beyond train_steps={args.train_steps}; "
            "skipping update loop and saving checkpoint."
        )
    for step in range(start_step, args.train_steps + 1):
        completed_step = step
        rng_key, collect_key = jax.random.split(rng_key)
        if use_pmap:
            collect_keys = jax.random.split(collect_key, local_devices)
            collect_temperature = jnp.full((local_devices,), jnp.float32(args.temperature), dtype=jnp.float32)
            obs, policy, value, mask, _, winners = collect_step(
                params,
                collect_keys,
                collect_temperature,
            )
        else:
            obs, policy, value, mask, _, winners = collect_step(
                params,
                collect_key,
                jnp.float32(args.temperature),
            )

        obs_np, policy_np, value_np, mask_np, winners_np = jax.device_get((obs, policy, value, mask, winners))
        obs_np = np.asarray(obs_np)
        policy_np = np.asarray(policy_np)
        value_np = np.asarray(value_np)
        mask_np = np.asarray(mask_np)
        winners_np = np.asarray(winners_np)

        flat_obs = obs_np.reshape((-1, args.board_size, args.board_size, 4))
        flat_policy = policy_np.reshape((-1, args.board_size * args.board_size))
        flat_value = value_np.reshape((-1,))
        valid = mask_np.reshape((-1,)).astype(bool)

        new_obs = flat_obs[valid]
        new_policy = flat_policy[valid]
        new_value = flat_value[valid]
        replay_obs, replay_policy, replay_value = _append_replay(
            replay_obs,
            replay_policy,
            replay_value,
            new_obs,
            new_policy,
            new_value,
            replay_size=args.replay_size,
        )

        if replay_obs.shape[0] < args.batch_size:
            print(f"step={step} replay={replay_obs.shape[0]} waiting for enough samples")
            maybe_save_training_checkpoint(step, "replay-wait")
            continue

        loss_sum = 0.0
        policy_sum = 0.0
        value_sum = 0.0
        for _ in range(args.updates_per_step):
            sample_ids = np_rng.integers(0, replay_obs.shape[0], size=args.batch_size)
            obs_batch = replay_obs[sample_ids]
            policy_batch = replay_policy[sample_ids]
            value_batch = replay_value[sample_ids]

            if not args.disable_symmetry_augmentation:
                obs_batch, policy_batch = _augment_batch_with_random_symmetry(
                    obs_batch,
                    policy_batch,
                    board_size=args.board_size,
                    np_rng=np_rng,
                )

            obs = jnp.asarray(obs_batch, dtype=jnp.float32)
            policy_target = jnp.asarray(policy_batch, dtype=jnp.float32)
            value_target = jnp.asarray(value_batch, dtype=jnp.float32)

            if use_pmap:
                obs = obs.reshape((local_devices, per_device_batch, args.board_size, args.board_size, 4))
                policy_target = policy_target.reshape((local_devices, per_device_batch, -1))
                value_target = value_target.reshape((local_devices, per_device_batch))
                params, opt_state, loss, policy_loss, value_loss = train_step(
                    params,
                    opt_state,
                    obs,
                    policy_target,
                    value_target,
                )
                loss_sum += float(loss[0])
                policy_sum += float(policy_loss[0])
                value_sum += float(value_loss[0])
            else:
                params, opt_state, loss, policy_loss, value_loss = train_step(
                    params,
                    opt_state,
                    obs,
                    policy_target,
                    value_target,
                )
                loss_sum += float(loss)
                policy_sum += float(policy_loss)
                value_sum += float(value_loss)
            optimizer_updates += 1

        loss_val = loss_sum / args.updates_per_step
        pol_val = policy_sum / args.updates_per_step
        val_val = value_sum / args.updates_per_step
        lr_val = float(lr_schedule(jnp.asarray(max(optimizer_updates - 1, 0), dtype=jnp.int32)))

        black_win = int((winners_np == 1).sum())
        white_win = int((winners_np == -1).sum())
        draw = int((winners_np == 0).sum())
        new_examples = int(valid.sum())

        print(
            f"step={step} lr={lr_val:.6f} loss={loss_val:.4f} policy={pol_val:.4f} value={val_val:.4f} "
            f"replay={replay_obs.shape[0]} new_examples={new_examples} "
            f"bw={black_win} ww={white_win} d={draw}"
        )

        if args.arena_every_steps > 0 and (step % args.arena_every_steps == 0):
            rng_key, arena_key = jax.random.split(rng_key)
            if use_pmap:
                arena_keys = jax.random.split(arena_key, local_devices)
                arena_wins, arena_losses, arena_draws = pmap_arena_step(params, best_params_repl, arena_keys)
                arena_wins = int(np.asarray(jax.device_get(arena_wins)).sum())
                arena_losses = int(np.asarray(jax.device_get(arena_losses)).sum())
                arena_draws = int(np.asarray(jax.device_get(arena_draws)).sum())
                current_params = _extract_host_tree(params, use_pmap=use_pmap)
            else:
                current_params = _extract_host_tree(params, use_pmap=use_pmap)
                arena_wins, arena_losses, arena_draws = arena_fn(current_params, best_params, arena_key)
                arena_wins = int(arena_wins)
                arena_losses = int(arena_losses)
                arena_draws = int(arena_draws)
            arena_win_rate = arena_wins / args.arena_games
            print(
                f"arena step={step} candidate_w={arena_wins} candidate_l={arena_losses} "
                f"draw={arena_draws} win_rate={arena_win_rate:.3f} threshold={args.arena_replace_threshold:.3f}"
            )
            if arena_win_rate >= args.arena_replace_threshold:
                best_params = current_params
                if use_pmap:
                    best_params_repl = _replicate_tree_for_pmap(best_params, local_devices=local_devices)
                best_step = step
                best_cfg = _checkpoint_config(args, optimizer_updates=optimizer_updates, best_step=best_step)
                _save_model_checkpoint(best_path, best_params, best_cfg)
                print(f"arena promoted candidate at step={step} -> {best_path}")

        maybe_save_training_checkpoint(step, "periodic")

    final_params = _extract_host_tree(params, use_pmap=use_pmap)
    final_opt_state = _extract_host_tree(opt_state, use_pmap=use_pmap)
    cfg = _checkpoint_config(args, optimizer_updates=optimizer_updates, best_step=best_step)
    _save_training_checkpoint(
        args.output,
        params=final_params,
        opt_state=final_opt_state,
        config=cfg,
        step=completed_step,
        optimizer_updates=optimizer_updates,
        rng_key=rng_key,
        np_rng_state=np_rng.bit_generator.state,
        replay_obs=replay_obs,
        replay_policy=replay_policy,
        replay_value=replay_value,
        best_params=best_params,
        best_step=best_step,
    )
    print(f"saved training checkpoint to {args.output}")
    _save_model_checkpoint(best_path, best_params, cfg)
    print(f"saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
