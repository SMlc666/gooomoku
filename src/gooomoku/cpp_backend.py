from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gooomoku import env
from gooomoku.net import PolicyValueNet

try:
    import gooomoku_cpp as _gooomoku_cpp
except Exception:
    _gooomoku_cpp = None


def is_cpp_backend_available() -> bool:
    return _gooomoku_cpp is not None


def _require_cpp_backend() -> None:
    if _gooomoku_cpp is None:
        raise RuntimeError(
            "gooomoku_cpp extension is not available. Build it with: "
            "python setup.py build_ext --inplace"
        )


def _as_numpy(x: Any, *, dtype=None) -> np.ndarray:
    arr = np.asarray(jax.device_get(x))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def gomoku_state_to_numpy_dict(states: env.GomokuState | dict[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(states, dict):
        out = {
            "board": np.asarray(states["board"], dtype=np.int8),
            "black_words": np.asarray(states["black_words"], dtype=np.uint32),
            "white_words": np.asarray(states["white_words"], dtype=np.uint32),
            "to_play": np.asarray(states["to_play"], dtype=np.int8),
            "last_action": np.asarray(states["last_action"], dtype=np.int32),
            "num_moves": np.asarray(states["num_moves"], dtype=np.int32),
            "terminated": np.asarray(states["terminated"], dtype=np.bool_),
            "winner": np.asarray(states["winner"], dtype=np.int8),
            "rule_flags": np.asarray(states["rule_flags"], dtype=np.int8),
            "swap_source_flag": np.asarray(states["swap_source_flag"], dtype=np.int8),
            "swap_applied_flag": np.asarray(states["swap_applied_flag"], dtype=np.int8),
        }
    else:
        out = {
            "board": _as_numpy(states.board, dtype=np.int8),
            "black_words": _as_numpy(states.black_words, dtype=np.uint32),
            "white_words": _as_numpy(states.white_words, dtype=np.uint32),
            "to_play": _as_numpy(states.to_play, dtype=np.int8),
            "last_action": _as_numpy(states.last_action, dtype=np.int32),
            "num_moves": _as_numpy(states.num_moves, dtype=np.int32),
            "terminated": _as_numpy(states.terminated, dtype=np.bool_),
            "winner": _as_numpy(states.winner, dtype=np.int8),
            "rule_flags": _as_numpy(states.rule_flags, dtype=np.int8),
            "swap_source_flag": _as_numpy(states.swap_source_flag, dtype=np.int8),
            "swap_applied_flag": _as_numpy(states.swap_applied_flag, dtype=np.int8),
        }

    board = out["board"]
    if board.ndim == 2:
        out = {k: v[None, ...] for k, v in out.items()}
    return out


def numpy_dict_to_gomoku_state(states: dict[str, np.ndarray]) -> env.GomokuState:
    return env.GomokuState(
        board=jnp.asarray(states["board"], dtype=jnp.int8),
        black_words=jnp.asarray(states["black_words"], dtype=jnp.uint32),
        white_words=jnp.asarray(states["white_words"], dtype=jnp.uint32),
        to_play=jnp.asarray(states["to_play"], dtype=jnp.int8),
        last_action=jnp.asarray(states["last_action"], dtype=jnp.int32),
        num_moves=jnp.asarray(states["num_moves"], dtype=jnp.int32),
        terminated=jnp.asarray(states["terminated"], dtype=jnp.bool_),
        winner=jnp.asarray(states["winner"], dtype=jnp.int8),
        rule_flags=jnp.asarray(states["rule_flags"], dtype=jnp.int8),
        swap_source_flag=jnp.asarray(states["swap_source_flag"], dtype=jnp.int8),
        swap_applied_flag=jnp.asarray(states["swap_applied_flag"], dtype=jnp.int8),
    )


def rng_key_to_seed(rng_key: Any) -> int:
    key_np = np.asarray(jax.device_get(rng_key), dtype=np.uint32).reshape(-1)
    if key_np.size == 0:
        return 0
    # Deterministic collapse to uint64 seed.
    seed = np.uint64(1469598103934665603)
    for v in key_np:
        seed ^= np.uint64(v)
        seed *= np.uint64(1099511628211)
    return int(seed & np.uint64((1 << 63) - 1))


@dataclass
class CppPolicyOutput:
    action_weights: jnp.ndarray
    selected_actions: jnp.ndarray
    root_values: jnp.ndarray


class CppSearchEngine:
    def __init__(
        self,
        *,
        model: PolicyValueNet,
        board_size: int,
        leaf_eval_batch_size: int,
        num_threads: int,
        virtual_loss: float,
        c_puct: float,
    ):
        _require_cpp_backend()
        self.model = model
        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.leaf_eval_batch_size = int(max(1, leaf_eval_batch_size))
        self.num_threads = int(num_threads)
        self.virtual_loss = float(max(0.0, virtual_loss))
        self.c_puct = float(c_puct)
        self.backend = _gooomoku_cpp.GomokuBackend(self.board_size)

        @jax.jit
        def _infer(params, obs):
            logits, value = self.model.apply(params, obs)
            return logits.astype(jnp.float32), value.astype(jnp.float32)

        self._infer = _infer

    def _eval_obs(self, params, obs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs_np = np.asarray(obs_np, dtype=np.float32)
        n = int(obs_np.shape[0])
        if n == 0:
            return (
                np.zeros((0, self.num_actions), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        logits_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        for start in range(0, n, self.leaf_eval_batch_size):
            end = min(start + self.leaf_eval_batch_size, n)
            chunk = obs_np[start:end]
            cur_n = int(chunk.shape[0])
            if cur_n < self.leaf_eval_batch_size:
                pad = np.zeros(
                    (self.leaf_eval_batch_size, self.board_size, self.board_size, env.OBS_PLANES),
                    dtype=np.float32,
                )
                pad[:cur_n] = chunk
                chunk = pad

            logits, value = self._infer(params, jnp.asarray(chunk, dtype=jnp.float32))
            logits_np, value_np = jax.device_get((logits, value))
            logits_chunks.append(np.asarray(logits_np[:cur_n], dtype=np.float32))
            value_chunks.append(np.asarray(value_np[:cur_n], dtype=np.float32))

        return np.concatenate(logits_chunks, axis=0), np.concatenate(value_chunks, axis=0)

    def search(
        self,
        *,
        params,
        states: env.GomokuState | dict[str, Any],
        root_prior_logits: jnp.ndarray | np.ndarray,
        root_values: jnp.ndarray | np.ndarray,
        rng_key,
        num_simulations: int,
        max_num_considered_actions: int,
        gumbel_scale: float,
        c_lcb: float,
    ) -> CppPolicyOutput:
        state_np = gomoku_state_to_numpy_dict(states)
        root_prior_np = np.asarray(jax.device_get(root_prior_logits), dtype=np.float32)
        root_values_np = np.asarray(jax.device_get(root_values), dtype=np.float32).reshape((-1,))
        seed = rng_key_to_seed(rng_key)

        def evaluator(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return self._eval_obs(params, obs_batch)

        result = self.backend.search_gumbel(
            state_np,
            root_prior_np,
            root_values_np,
            evaluator,
            int(num_simulations),
            int(max_num_considered_actions),
            float(gumbel_scale),
            float(self.c_puct),
            float(c_lcb),
            float(self.virtual_loss),
            int(self.leaf_eval_batch_size),
            int(self.num_threads),
            int(seed),
        )

        return CppPolicyOutput(
            action_weights=jnp.asarray(np.asarray(result["action_weights"], dtype=np.float32)),
            selected_actions=jnp.asarray(np.asarray(result["selected_actions"], dtype=np.int32)),
            root_values=jnp.asarray(np.asarray(result["root_values"], dtype=np.float32)),
        )

    def root_policy(
        self,
        *,
        params,
        states: env.GomokuState | dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        state_np = gomoku_state_to_numpy_dict(states)
        obs = np.asarray(self.backend.batch_encode(state_np), dtype=np.float32)
        logits, value = self._eval_obs(params, obs)
        legal = np.asarray(self.backend.batch_legal_mask(state_np), dtype=np.bool_)
        min_logit = np.finfo(np.float32).min
        prior_logits = np.where(legal, logits, min_logit).astype(np.float32, copy=False)
        return state_np, prior_logits, value.astype(np.float32, copy=False), legal

    def batch_encode(self, states: env.GomokuState | dict[str, Any]) -> np.ndarray:
        return np.asarray(self.backend.batch_encode(gomoku_state_to_numpy_dict(states)), dtype=np.float32)

    def batch_step(
        self,
        states: env.GomokuState | dict[str, Any],
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        next_state, reward, done = self.backend.batch_step(
            gomoku_state_to_numpy_dict(states),
            np.asarray(actions, dtype=np.int32),
        )
        return gomoku_state_to_numpy_dict(next_state), np.asarray(reward, dtype=np.float32), np.asarray(done, dtype=np.bool_)
