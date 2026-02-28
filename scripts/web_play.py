from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import hashlib
import os
import pickle
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

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


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"unsupported tar member type: {member.name}")
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_dir.resolve())):
                raise ValueError(f"unsafe tar member path: {member.name}")
            if member.isfile() or member.isdir():
                tf.extract(member, path=target_dir, set_attrs=False)


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_artifact_sha256(path: Path, expected_sha256: str) -> None:
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError("artifact-sha256 must be 64 hex chars")
    actual = _sha256_of_file(path)
    if actual != normalized:
        raise ValueError(f"artifact sha256 mismatch: expected={normalized} actual={actual}")


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"invalid checkpoint payload in {path}")
    return payload


def _resolve_artifact_checkpoint(artifact_path: Path, expected_sha256: str | None) -> tuple[dict[str, Any], str]:
    if expected_sha256:
        _verify_artifact_sha256(artifact_path, expected_sha256)

    if artifact_path.suffix == ".pkl":
        return _load_checkpoint_payload(artifact_path), str(artifact_path)

    suffixes = artifact_path.suffixes
    is_tar_gz = len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".gz"]
    is_tgz = suffixes[-1:] == [".tgz"]
    if not (is_tar_gz or is_tgz):
        raise ValueError("model artifact must be .pkl, .tar.gz, or .tgz")

    with tempfile.TemporaryDirectory(prefix="gooomoku_model_") as tmp:
        tmp_path = Path(tmp)
        _safe_extract_tar(artifact_path, tmp_path)
        candidates = sorted(tmp_path.glob("**/*.pkl"))
        if not candidates:
            raise ValueError(f"no .pkl checkpoint found inside {artifact_path}")
        checkpoint_path = candidates[0]
        payload = _load_checkpoint_payload(checkpoint_path)
        return payload, str(checkpoint_path)


def _add_batch_dim(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _state_to_board_list(state: env.GomokuState) -> list[list[int]]:
    return jax.device_get(state.board).astype(int).tolist()


def _winner_to_status(winner: int) -> str:
    if winner == 1:
        return "black_win"
    if winner == -1:
        return "white_win"
    return "draw"


class NewGameResponse(BaseModel):
    board: list[list[int]]
    history: list[int]
    to_play: int
    status: str
    winner: int
    ai_action: int | None = None


class MoveRequest(BaseModel):
    history: list[int] = Field(default_factory=list)
    human_action: int = Field(ge=0)
    human_color: int = Field(default=1)


class MoveResponse(BaseModel):
    board: list[list[int]]
    history: list[int]
    to_play: int
    terminated: bool
    status: str
    winner: int
    human_action: int
    ai_action: int | None = None


class GameEngine:
    def __init__(
        self,
        *,
        params,
        model: PolicyValueNet,
        board_size: int,
        num_simulations: int,
        max_num_considered_actions: int,
        ai_temperature: float,
        c_lcb: float,
        seed: int,
    ):
        self.params = params
        self.model = model
        self.board_size = board_size
        self.num_actions = board_size * board_size
        self.search_fn = build_search_fn(
            model=self.model,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            root_dirichlet_fraction=0.0,
            root_dirichlet_alpha=0.03,
            c_lcb=c_lcb,
        )
        self.rng_key = jax.random.PRNGKey(seed)
        self._lock = threading.Lock()
        self._warmup()

    def _warmup(self) -> None:
        dummy = env.reset(self.board_size)
        with self._lock:
            self.rng_key, run_key = jax.random.split(self.rng_key)
            out = self.search_fn(self.params, _add_batch_dim(dummy), run_key)
            out.action_weights.block_until_ready()

    def _replay_history(self, history: list[int]) -> env.GomokuState:
        state = env.reset(self.board_size)
        for action in history:
            if action < 0 or action >= self.num_actions:
                raise HTTPException(status_code=400, detail=f"history contains invalid action={action}")
            state, _, _ = env.step(state, jnp.int32(action))
            if bool(jax.device_get(state.terminated)):
                break
        return state

    def _pick_ai_action(self, state: env.GomokuState) -> int:
        with self._lock:
            self.rng_key, search_key, sample_key = jax.random.split(self.rng_key, 3)
            policy_output = self.search_fn(self.params, _add_batch_dim(state), search_key)
            visit_probs = policy_output.action_weights[0]
            if self.ai_temperature <= 1e-6:
                action = jnp.argmax(visit_probs).astype(jnp.int32)
            else:
                logits = jnp.log(jnp.maximum(visit_probs, 1e-8)) / jnp.float32(self.ai_temperature)
                action = jax.random.categorical(sample_key, logits).astype(jnp.int32)
            return int(jax.device_get(action))

    def new_game(self, human_color: int) -> NewGameResponse:
        state = env.reset(self.board_size)
        history: list[int] = []
        ai_action: int | None = None

        if human_color == -1:
            ai_action = self._pick_ai_action(state)
            state, _, _ = env.step(state, jnp.int32(ai_action))
            history.append(ai_action)

        terminated = bool(jax.device_get(state.terminated))
        winner = int(jax.device_get(state.winner))
        return NewGameResponse(
            board=_state_to_board_list(state),
            history=history,
            to_play=int(jax.device_get(state.to_play)),
            status="ongoing" if not terminated else _winner_to_status(winner),
            winner=winner,
            ai_action=ai_action,
        )

    def play_move(self, req: MoveRequest) -> MoveResponse:
        if req.human_color not in (-1, 1):
            raise HTTPException(status_code=400, detail="human_color must be 1 (black) or -1 (white)")

        state = self._replay_history(req.history)
        if bool(jax.device_get(state.terminated)):
            raise HTTPException(status_code=400, detail="game already terminated")

        current_to_play = int(jax.device_get(state.to_play))
        if current_to_play != req.human_color:
            raise HTTPException(status_code=400, detail="not human turn")

        if req.human_action >= self.num_actions:
            raise HTTPException(status_code=400, detail=f"human_action must be < {self.num_actions}")

        legal = env.legal_action_mask(state)
        if not bool(jax.device_get(legal[req.human_action])):
            raise HTTPException(status_code=400, detail="illegal human action")

        history = list(req.history)
        state, _, _ = env.step(state, jnp.int32(req.human_action))
        history.append(req.human_action)
        human_terminated = bool(jax.device_get(state.terminated))
        ai_action: int | None = None

        if not human_terminated:
            ai_action = self._pick_ai_action(state)
            state, _, _ = env.step(state, jnp.int32(ai_action))
            history.append(ai_action)

        terminated = bool(jax.device_get(state.terminated))
        winner = int(jax.device_get(state.winner))
        return MoveResponse(
            board=_state_to_board_list(state),
            history=history,
            to_play=int(jax.device_get(state.to_play)),
            terminated=terminated,
            status="ongoing" if not terminated else _winner_to_status(winner),
            winner=winner,
            human_action=req.human_action,
            ai_action=ai_action,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gomoku web PvAI service with JAX GPU inference.")
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=256)
    parser.add_argument("--max-num-considered-actions", type=int, default=64)
    parser.add_argument("--compute-dtype", type=str, default="bfloat16")
    parser.add_argument("--param-dtype", type=str, default="float32")
    parser.add_argument("--ai-temperature", type=float, default=0.0)
    parser.add_argument("--c-lcb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jax-platforms", type=str, default="")
    parser.add_argument("--artifact-sha256", type=str, default="")
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    if args.jax_platforms:
        jax.config.update("jax_platforms", args.jax_platforms)
    configure_jax_runtime(app_name="web_play", repo_root=REPO_ROOT)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    payload, _resolved_path = _resolve_artifact_checkpoint(args.model_artifact, args.artifact_sha256)
    config = payload.get("config", {})

    board_size = int(config.get("board_size", args.board_size))
    channels = int(config.get("channels", args.channels))
    blocks = int(config.get("blocks", args.blocks))
    compute_dtype_name = str(config.get("compute_dtype", args.compute_dtype))
    param_dtype_name = str(config.get("param_dtype", args.param_dtype))
    num_simulations = int(config.get("num_simulations", args.num_simulations))
    max_num_considered_actions = int(config.get("max_num_considered_actions", args.max_num_considered_actions))

    compute_dtype = _dtype_from_name(compute_dtype_name)
    param_dtype = _dtype_from_name(param_dtype_name)

    model = PolicyValueNet(
        board_size=board_size,
        channels=channels,
        blocks=blocks,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
    )

    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    engine = GameEngine(
        params=params,
        model=model,
        board_size=board_size,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        ai_temperature=args.ai_temperature,
        c_lcb=args.c_lcb,
        seed=args.seed,
    )

    app = FastAPI(title="gooomoku web PvAI")
    static_dir = REPO_ROOT / "web"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device_count": jax.local_device_count(),
            "devices": [str(d) for d in jax.devices()],
            "board_size": board_size,
            "channels": channels,
            "blocks": blocks,
            "num_simulations": num_simulations,
            "max_num_considered_actions": max_num_considered_actions,
            "model_artifact": str(args.model_artifact),
            "c_lcb": args.c_lcb,
        }
    @app.post("/api/new", response_model=NewGameResponse)
    def new_game(human_color: int = 1) -> NewGameResponse:
        if human_color not in (-1, 1):
            raise HTTPException(status_code=400, detail="human_color must be 1 (black) or -1 (white)")
        return engine.new_game(human_color)

    @app.post("/api/move", response_model=MoveResponse)
    def move(req: MoveRequest) -> MoveResponse:
        return engine.play_move(req)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
