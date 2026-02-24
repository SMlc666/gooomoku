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

from gooomoku.net import PolicyValueNet
from scripts.self_play import build_play_many_games_fn


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
            logits, value = model.apply(trainable, obs)
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
            logits, value = model.apply(trainable, obs)
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


def _save_checkpoint(path: Path, params, config: dict) -> None:
    payload = {"params": jax.device_get(params), "config": config}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(payload, fp)


def _extract_host_params(params, use_pmap: bool):
    return jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params


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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--max-num-considered-actions", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=12)
    parser.add_argument("--final-temperature", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--disable-symmetry-augmentation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest.pkl"))
    parser.add_argument("--disable-pmap", action="store_true")
    args = parser.parse_args()

    if args.updates_per_step < 1:
        raise ValueError("updates-per-step must be >= 1")

    model = PolicyValueNet(board_size=args.board_size, channels=args.channels, blocks=args.blocks)
    rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = model.init(init_key, jnp.zeros((1, args.board_size, args.board_size, 4), dtype=jnp.float32))

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
    local_devices = jax.local_device_count()
    use_pmap = (not args.disable_pmap) and local_devices > 1
    if use_pmap and (args.batch_size % local_devices != 0):
        raise ValueError(f"batch-size must be divisible by local_device_count={local_devices}")
    if use_pmap and (args.games_per_step % local_devices != 0):
        raise ValueError(f"games-per-step must be divisible by local_device_count={local_devices}")

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

    if use_pmap:
        per_device_batch = args.batch_size // local_devices
        train_step = make_pmap_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = make_pmap_collect_step(play_many_games_fn)
        params = _replicate_tree_for_pmap(params, local_devices=local_devices)
        opt_state = _replicate_tree_for_pmap(opt_state, local_devices=local_devices)
        print(
            f"pmap enabled: local_device_count={local_devices}, per_device_batch={per_device_batch}, "
            f"selfplay_games_per_device={games_per_device}, updates_per_step={args.updates_per_step}"
        )
    else:
        train_step = make_single_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = play_many_games_fn
        per_device_batch = args.batch_size
        print(
            f"single-device mode: local_device_count={local_devices}, updates_per_step={args.updates_per_step}"
        )

    replay_obs = np.zeros((0, args.board_size, args.board_size, 4), dtype=np.float32)
    replay_policy = np.zeros((0, args.board_size * args.board_size), dtype=np.float32)
    replay_value = np.zeros((0,), dtype=np.float32)
    np_rng = np.random.default_rng(args.seed + 13)
    optimizer_updates = 0

    for step in range(1, args.train_steps + 1):
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

    final_params = _extract_host_params(params, use_pmap=use_pmap)
    cfg = {
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
        "symmetry_augmentation": not args.disable_symmetry_augmentation,
    }
    _save_checkpoint(args.output, final_params, cfg)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
