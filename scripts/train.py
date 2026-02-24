from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from gooomoku.net import PolicyValueNet
from scripts.self_play import TrainingExample, play_one_game, stack_examples


def _l2_regularization(params) -> jnp.ndarray:
    return sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params))


def make_single_train_step(model: PolicyValueNet, optimizer: optax.GradientTransformation, weight_decay: float):
    @jax.jit
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
    @jax.pmap(axis_name="device")
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


def _save_checkpoint(path: Path, params, config: dict) -> None:
    payload = {"params": jax.device_get(params), "config": config}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(payload, fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal JAX+mctx gomoku trainer.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--games-per-step", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--max-num-considered-actions", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest.pkl"))
    parser.add_argument("--disable-pmap", action="store_true")
    args = parser.parse_args()

    model = PolicyValueNet(board_size=args.board_size, channels=args.channels, blocks=args.blocks)
    rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = model.init(init_key, jnp.zeros((1, args.board_size, args.board_size, 4), dtype=jnp.float32))

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    local_devices = jax.local_device_count()
    use_pmap = (not args.disable_pmap) and local_devices > 1
    if use_pmap and (args.batch_size % local_devices != 0):
        raise ValueError(f"batch-size must be divisible by local_device_count={local_devices}")

    if use_pmap:
        per_device_batch = args.batch_size // local_devices
        train_step = make_pmap_train_step(model, optimizer, weight_decay=args.weight_decay)
        devices = jax.local_devices()
        params = jax.device_put_replicated(params, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)
        print(f"pmap enabled: local_device_count={local_devices}, per_device_batch={per_device_batch}")
    else:
        train_step = make_single_train_step(model, optimizer, weight_decay=args.weight_decay)
        per_device_batch = args.batch_size
        print(f"single-device mode: local_device_count={local_devices}")

    replay: List[TrainingExample] = []
    np_rng = np.random.default_rng(args.seed + 13)

    for step in range(1, args.train_steps + 1):
        stats = {"black_win": 0, "white_win": 0, "draw": 0, "examples": 0}
        current_params = jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params

        for _ in range(args.games_per_step):
            rng_key, game_key = jax.random.split(rng_key)
            examples, winner = play_one_game(
                params=current_params,
                model=model,
                rng_key=game_key,
                board_size=args.board_size,
                num_simulations=args.num_simulations,
                max_num_considered_actions=args.max_num_considered_actions,
                temperature=args.temperature,
            )
            replay.extend(examples)
            stats["examples"] += len(examples)
            if winner == 1:
                stats["black_win"] += 1
            elif winner == -1:
                stats["white_win"] += 1
            else:
                stats["draw"] += 1

        if len(replay) > args.replay_size:
            replay = replay[-args.replay_size :]

        if len(replay) < args.batch_size:
            print(f"step={step} replay={len(replay)} waiting for enough samples")
            continue

        sample_ids = np_rng.integers(0, len(replay), size=args.batch_size)
        batch = [replay[idx] for idx in sample_ids]
        obs, policy_target, value_target = stack_examples(batch)

        if use_pmap:
            obs = obs.reshape((local_devices, per_device_batch, args.board_size, args.board_size, 4))
            policy_target = policy_target.reshape((local_devices, per_device_batch, -1))
            value_target = value_target.reshape((local_devices, per_device_batch))
            params, opt_state, loss, policy_loss, value_loss = train_step(params, opt_state, obs, policy_target, value_target)
            loss_val = float(loss[0])
            pol_val = float(policy_loss[0])
            val_val = float(value_loss[0])
        else:
            params, opt_state, loss, policy_loss, value_loss = train_step(params, opt_state, obs, policy_target, value_target)
            loss_val = float(loss)
            pol_val = float(policy_loss)
            val_val = float(value_loss)

        print(
            f"step={step} loss={loss_val:.4f} policy={pol_val:.4f} value={val_val:.4f} "
            f"replay={len(replay)} new_examples={stats['examples']} "
            f"bw={stats['black_win']} ww={stats['white_win']} d={stats['draw']}"
        )

    final_params = jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params
    cfg = {
        "board_size": args.board_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "num_simulations": args.num_simulations,
        "max_num_considered_actions": args.max_num_considered_actions,
    }
    _save_checkpoint(args.output, final_params, cfg)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
