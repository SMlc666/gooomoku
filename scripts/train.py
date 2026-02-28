from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import pickle
import queue as queue_lib
import socket
import struct
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import threading
import time
import tempfile
from typing import Any

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
from gooomoku.replay_wire import ReplayWireError
from gooomoku.replay_wire import connect_with_retry
from gooomoku.replay_wire import recv_selfplay_batch
from gooomoku.replay_wire import send_selfplay_batch
from gooomoku.runtime import configure_jax_runtime
from scripts.self_play import build_play_many_games_fn

SelfPlayBatch = tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]
ActorError = tuple[int, BaseException]
ActorMessage = tuple[str, Any]


def _looks_like_tpu_runtime_error(exc: BaseException) -> bool:
    text = str(exc)
    patterns = (
        "TPU is already in use by process",
        "Unable to initialize backend 'tpu'",
        "initialize_pjrt_plugin('tpu')",
        "open(/dev/accel0)",
        "Couldn't open device: /dev/accel0",
    )
    return any(p in text for p in patterns)


def _print_tpu_startup_diagnostics(*, stage: str, exc: BaseException) -> None:
    detail = str(exc)
    diag_lines = [
        "[gooomoku][tpu-startup] hard-fail (no fallback)",
        f"TPU runtime initialization failed during {stage}.",
        f"Original error: {detail}",
        f"pid={os.getpid()} JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '')!r}",
        f"TPU_WORKER_ID={os.environ.get('TPU_WORKER_ID', '')!r} CLOUD_TPU_TASK_ID={os.environ.get('CLOUD_TPU_TASK_ID', '')!r}",
        "Likely causes:",
        "  1) another process on this host already owns TPU (/tmp/libtpu_lockfile)",
        "  2) this worker cannot access /dev/accel0 (runtime/permission issue)",
        "Recommended checks:",
        "  - ensure exactly one trainer process per host",
        "  - inspect current TPU owner: sudo lsof -w /dev/accel0",
        "  - stop stale TPU owners shown in the error PID",
        "  - remove stale /tmp/libtpu_lockfile and /tmp/tpu_logs if previous run crashed",
        "  - verify TPU VM/runtime health: python -c \"import jax; print(jax.devices())\"",
    ]
    print("\n".join(diag_lines), file=sys.stderr)


def _run_cross_process_actor(
    actor_idx: int,
    cfg: dict[str, Any],
    params_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: Any,
) -> None:
    actor_jax_platforms = str(cfg.get("actor_jax_platforms", "cpu")).strip()
    if actor_jax_platforms:
        os.environ["JAX_PLATFORMS"] = actor_jax_platforms
        os.environ.pop("JAX_PLATFORM_NAME", None)
        jax.config.update("jax_platforms", actor_jax_platforms)
    configure_jax_runtime(app_name=f"train-actor-{actor_idx}", repo_root=REPO_ROOT)

    model = PolicyValueNet(
        board_size=int(cfg["board_size"]),
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        compute_dtype=_dtype_from_name(str(cfg["compute_dtype"])),
        param_dtype=_dtype_from_name(str(cfg["param_dtype"])),
    )
    play_many_games_fn = build_play_many_games_fn(
        model=model,
        board_size=int(cfg["board_size"]),
        num_simulations=int(cfg["num_simulations"]),
        max_num_considered_actions=int(cfg["max_num_considered_actions"]),
        num_games=int(cfg["num_games"]),
        temperature_drop_move=int(cfg["temperature_drop_move"]),
        final_temperature=float(cfg["final_temperature"]),
        root_dirichlet_fraction=float(cfg["root_dirichlet_fraction"]),
        root_dirichlet_alpha=float(cfg["root_dirichlet_alpha"]),
    )

    rng = jax.random.PRNGKey(int(cfg["seed"]) + 100003 * (actor_idx + 1))
    params = params_queue.get()
    board_size = int(cfg["board_size"])
    fixed_examples = int(cfg.get("replay_fixed_update_size", 0))
    temperature = jnp.float32(float(cfg["temperature"]))

    while not stop_event.is_set():
        try:
            while True:
                try:
                    params = params_queue.get_nowait()
                except queue_lib.Empty:
                    break

            rng, collect_key = jax.random.split(rng)
            obs, policy, value, mask, _, winners = play_many_games_fn(params, collect_key, temperature)
            obs_np, policy_np, value_np, mask_np, winners_np = jax.device_get((obs, policy, value, mask, winners))
            obs_np = np.asarray(obs_np)
            policy_np = np.asarray(policy_np)
            value_np = np.asarray(value_np)
            mask_np = np.asarray(mask_np)
            winners_np = np.asarray(winners_np)

            flat_obs = obs_np.reshape((-1, board_size, board_size, 4))
            flat_policy = policy_np.reshape((-1, board_size * board_size))
            flat_value = value_np.reshape((-1,))
            valid = mask_np.reshape((-1,)).astype(bool)

            packed_obs, packed_policy, packed_value, stored_examples = _stabilize_replay_payload_examples(
                flat_obs,
                flat_policy,
                flat_value,
                valid,
                fixed_examples=fixed_examples,
            )
            payload = (
                packed_obs,
                packed_policy,
                packed_value,
                int((winners_np == 1).sum()),
                int((winners_np == -1).sum()),
                int((winners_np == 0).sum()),
                stored_examples,
            )

            while not stop_event.is_set():
                try:
                    output_queue.put(("data", payload), timeout=0.2)
                    break
                except queue_lib.Full:
                    continue
        except BaseException as exc:
            output_queue.put(("error", (actor_idx, repr(exc))))
            break


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


def make_single_multi_train_step(model: PolicyValueNet, optimizer: optax.GradientTransformation, weight_decay: float):
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, obs_batches, policy_batches, value_batches):
        def one_update(carry, inputs):
            current_params, current_opt_state = carry
            obs, policy_target, value_target = inputs

            def loss_fn(trainable):
                logits, value = model.apply(trainable, obs.astype(model.compute_dtype))
                logits = logits.astype(jnp.float32)
                value = value.astype(jnp.float32)
                policy_loss = -jnp.mean(jnp.sum(policy_target * jax.nn.log_softmax(logits), axis=-1))
                value_loss = jnp.mean(jnp.square(value - value_target))
                reg = jnp.float32(weight_decay) * _l2_regularization(trainable)
                total_loss = policy_loss + value_loss + reg
                return total_loss, (policy_loss, value_loss)

            (loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            return (next_params, next_opt_state), (loss, policy_loss, value_loss)

        (new_params, new_opt_state), (losses, policy_losses, value_losses) = jax.lax.scan(
            one_update,
            (params, opt_state),
            (obs_batches, policy_batches, value_batches),
        )
        return (
            new_params,
            new_opt_state,
            jnp.mean(losses),
            jnp.mean(policy_losses),
            jnp.mean(value_losses),
        )

    return train_step


def make_pmap_multi_train_step(model: PolicyValueNet, optimizer: optax.GradientTransformation, weight_decay: float):
    @functools.partial(jax.pmap, axis_name="device", donate_argnums=(0, 1))
    def train_step(params, opt_state, obs_batches, policy_batches, value_batches):
        def one_update(carry, inputs):
            current_params, current_opt_state = carry
            obs, policy_target, value_target = inputs

            def loss_fn(trainable):
                logits, value = model.apply(trainable, obs.astype(model.compute_dtype))
                logits = logits.astype(jnp.float32)
                value = value.astype(jnp.float32)
                policy_loss = -jnp.mean(jnp.sum(policy_target * jax.nn.log_softmax(logits), axis=-1))
                value_loss = jnp.mean(jnp.square(value - value_target))
                reg = jnp.float32(weight_decay) * _l2_regularization(trainable)
                total_loss = policy_loss + value_loss + reg
                return total_loss, (policy_loss, value_loss)

            (loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
            grads = jax.lax.pmean(grads, axis_name="device")
            loss = jax.lax.pmean(loss, axis_name="device")
            policy_loss = jax.lax.pmean(policy_loss, axis_name="device")
            value_loss = jax.lax.pmean(value_loss, axis_name="device")
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            return (next_params, next_opt_state), (loss, policy_loss, value_loss)

        (new_params, new_opt_state), (losses, policy_losses, value_losses) = jax.lax.scan(
            one_update,
            (params, opt_state),
            (obs_batches, policy_batches, value_batches),
        )
        return (
            new_params,
            new_opt_state,
            jnp.mean(losses),
            jnp.mean(policy_losses),
            jnp.mean(value_losses),
        )

    return train_step


def make_pmap_collect_step(play_many_games_fn, *, devices=None):
    def collect_step(params, rng_key, temperature):
        return play_many_games_fn(params, rng_key, temperature)

    if devices is None:
        return jax.pmap(collect_step, axis_name="device")
    return jax.pmap(collect_step, axis_name="device", devices=tuple(devices))


def make_pmap_arena_step(arena_fn):
    @functools.partial(jax.pmap, axis_name="device")
    def arena_step(params_a, params_b, rng_key):
        return arena_fn(params_a, params_b, rng_key)

    return arena_step


def _is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def _gcs_object_path(path: str) -> tuple[str, str]:
    rest = path[5:]
    if "/" not in rest:
        raise ValueError(f"GCS path must include object name: {path}")
    bucket, obj = rest.split("/", 1)
    if not bucket or not obj:
        raise ValueError(f"invalid GCS path: {path}")
    return bucket, obj


def _best_checkpoint_path(path: str) -> str:
    if _is_gcs_uri(path):
        bucket, obj = _gcs_object_path(path)
        pure_obj = PurePosixPath(obj)
        suffix = pure_obj.suffix
        if suffix:
            best_name = f"{pure_obj.stem}.best{suffix}"
        else:
            best_name = f"{pure_obj.name}.best"
        best_obj = str(pure_obj.with_name(best_name))
        return f"gs://{bucket}/{best_obj}"
    local_path = Path(path)
    return str(local_path.parent / f"{local_path.stem}.best{local_path.suffix}")


def _write_pickle(path: str, payload: dict[str, Any]) -> None:
    if _is_gcs_uri(path):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pickle.dump(payload, tmp)
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", str(tmp_path), path],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("gcloud CLI is required for gs:// checkpoint writes") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as fp:
        pickle.dump(payload, fp)


def _read_pickle(path: str):
    if _is_gcs_uri(path):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", path, str(tmp_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("gcloud CLI is required for gs:// checkpoint reads") from exc
            with tmp_path.open("rb") as fp:
                return pickle.load(fp)
        finally:
            tmp_path.unlink(missing_ok=True)

    with Path(path).open("rb") as fp:
        return pickle.load(fp)


def _write_text(path: str, content: str) -> None:
    if _is_gcs_uri(path):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", str(tmp_path), path],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("gcloud CLI is required for gs:// text writes") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content, encoding="utf-8")


def _read_text(path: str) -> str:
    if _is_gcs_uri(path):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", path, str(tmp_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("gcloud CLI is required for gs:// text reads") from exc
            return tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)

    return Path(path).read_text(encoding="utf-8")


def _derive_replay_endpoint_uri(*, role: str, replay_endpoint_uri: str, output: str, resume_from: str | None) -> str | None:
    uri = replay_endpoint_uri.strip()
    if uri:
        return uri
    if role == "actor":
        anchor = resume_from or output
    else:
        anchor = output
    if not anchor:
        return None
    return f"{anchor}.replay_endpoint.json"


def _parse_tpu_worker_hosts_from_env() -> list[str]:
    raw = os.environ.get("TPU_WORKER_HOSTNAMES", "").strip()
    if not raw:
        return []
    hosts: list[str] = []
    for item in raw.split(","):
        host = item.strip()
        if not host:
            continue
        if ":" in host:
            host = host.split(":", 1)[0].strip()
        if host:
            hosts.append(host)
    return hosts


def _infer_learner_host_from_env() -> str | None:
    explicit = os.environ.get("REPLAY_LEARNER_HOST", "").strip()
    if explicit:
        return explicit

    worker_hosts = _parse_tpu_worker_hosts_from_env()
    if worker_hosts:
        return worker_hosts[0]

    coordinator = os.environ.get("MEGASCALE_COORDINATOR_ADDRESS", "").strip()
    if coordinator:
        return coordinator.split(":", 1)[0].strip() or None
    return None


def _detect_advertise_host() -> str:
    env_host = os.environ.get("REPLAY_ADVERTISE_HOST", "").strip()
    if env_host:
        return env_host
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))
            ip = probe.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            pass
    hostname = socket.gethostname().strip()
    if hostname:
        try:
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            pass
        return hostname
    return "127.0.0.1"


def _publish_replay_endpoint(path: str, host: str, port: int) -> None:
    payload = {
        "schema": "gooomoku-replay-endpoint-v1",
        "host": host,
        "port": int(port),
        "updated_at_unix": time.time(),
    }
    _write_text(path, json.dumps(payload, separators=(",", ":")))


def _load_replay_endpoint(path: str) -> tuple[str, int]:
    data = json.loads(_read_text(path))
    if data.get("schema") != "gooomoku-replay-endpoint-v1":
        raise RuntimeError(f"unsupported replay endpoint schema in {path}: {data.get('schema')}")
    host = str(data.get("host", "")).strip()
    port = int(data.get("port", 0))
    if not host:
        raise RuntimeError(f"missing host in replay endpoint file: {path}")
    if host in {"0.0.0.0", "::"}:
        raise RuntimeError(f"replay endpoint file contains wildcard host: {host}")
    if port <= 0:
        raise RuntimeError(f"invalid port in replay endpoint file: {path}")
    return host, port


def _resolve_actor_replay_endpoint(args) -> tuple[str, int]:
    host = args.replay_host.strip()
    if host and host.lower() != "auto":
        if host in {"0.0.0.0", "::"}:
            raise ValueError("--replay-host cannot be a wildcard address for --role actor; use learner hostname/IP")
        return host, args.replay_port

    inferred_host = _infer_learner_host_from_env()
    if inferred_host:
        print(f"actor endpoint inferred from TPU env: {inferred_host}:{args.replay_port}")
        return inferred_host, args.replay_port

    endpoint_uri = _derive_replay_endpoint_uri(
        role="actor",
        replay_endpoint_uri=args.replay_endpoint_uri,
        output=args.output,
        resume_from=args.resume_from,
    )
    if endpoint_uri is None:
        raise ValueError(
            "actor auto-discovery requires TPU worker env (TPU_WORKER_HOSTNAMES/REPLAY_LEARNER_HOST) "
            "or --replay-endpoint-uri / --resume-from / --output as discovery anchor"
        )
    wait_deadline = time.time() + max(float(args.replay_endpoint_wait_seconds), 0.1)
    retry_sleep = max(float(args.replay_connect_retry_seconds), 0.1)
    last_error: BaseException | None = None
    while True:
        try:
            resolved_host, resolved_port = _load_replay_endpoint(endpoint_uri)
            print(f"actor endpoint resolved from {endpoint_uri}: {resolved_host}:{resolved_port}")
            return resolved_host, resolved_port
        except BaseException as exc:  # retry until learner publishes endpoint
            last_error = exc
        if time.time() >= wait_deadline:
            raise RuntimeError(
                f"failed to resolve learner replay endpoint from {endpoint_uri} within "
                f"{args.replay_endpoint_wait_seconds}s: {last_error}"
            )
        time.sleep(retry_sleep)


_PARAM_FRAME_LEN_STRUCT = struct.Struct("!I")


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_PARAM_FRAME_LEN_STRUCT.pack(len(payload)))
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("socket closed while receiving frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> bytes:
    (size,) = _PARAM_FRAME_LEN_STRUCT.unpack(_recv_exact(sock, _PARAM_FRAME_LEN_STRUCT.size))
    if size < 0:
        raise RuntimeError("invalid negative frame size")
    return _recv_exact(sock, size) if size > 0 else b""


def _build_param_sync_blob(*, params, use_pmap: bool, optimizer_updates: int) -> bytes:
    params_host = _extract_host_tree(params, use_pmap=use_pmap)
    params_host = jax.device_get(params_host)
    payload = {
        "schema": "gooomoku-param-sync-v1",
        "optimizer_updates": int(optimizer_updates),
        "params": params_host,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _start_param_sync_server(*, host: str, port: int, snapshot_state: dict[str, Any], snapshot_lock: threading.Lock):
    stop_event = threading.Event()
    errors: list[BaseException] = []
    client_threads: list[threading.Thread] = []
    client_threads_lock = threading.Lock()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    server_sock.settimeout(1.0)

    def client_loop(conn: socket.socket, addr: tuple[str, int]) -> None:
        with conn:
            try:
                request = json.loads(_recv_frame(conn).decode("utf-8"))
                last_seen = int(request.get("last_optimizer_updates", -1))
                with snapshot_lock:
                    cur_updates = int(snapshot_state["optimizer_updates"])
                    cur_blob = snapshot_state["blob"]
                if cur_updates <= last_seen:
                    _send_frame(
                        conn,
                        json.dumps(
                            {
                                "status": "unchanged",
                                "optimizer_updates": cur_updates,
                            },
                            separators=(",", ":"),
                        ).encode("utf-8"),
                    )
                else:
                    _send_frame(
                        conn,
                        json.dumps(
                            {
                                "status": "update",
                                "optimizer_updates": cur_updates,
                            },
                            separators=(",", ":"),
                        ).encode("utf-8"),
                    )
                    _send_frame(conn, cur_blob)
            except BaseException as exc:
                if not stop_event.is_set():
                    errors.append(RuntimeError(f"param sync client error from {addr}: {exc}"))

    def accept_loop() -> None:
        while not stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not stop_event.is_set():
                    errors.append(RuntimeError(f"param sync accept failed: {exc}"))
                break
            worker = threading.Thread(target=client_loop, args=(conn, addr), daemon=True, name=f"param-sync-{addr[0]}:{addr[1]}")
            worker.start()
            with client_threads_lock:
                client_threads.append(worker)

    accept_thread = threading.Thread(target=accept_loop, daemon=True, name="param-sync-listener")
    accept_thread.start()
    return stop_event, errors, server_sock, accept_thread, client_threads, client_threads_lock


def _stop_param_sync_server(
    *,
    stop_event: threading.Event,
    server_sock: socket.socket | None,
    accept_thread: threading.Thread | None,
    client_threads: list[threading.Thread],
    client_threads_lock: threading.Lock,
) -> None:
    stop_event.set()
    if server_sock is not None:
        server_sock.close()
    if accept_thread is not None:
        accept_thread.join(timeout=5.0)
    with client_threads_lock:
        snapshot = list(client_threads)
    for worker in snapshot:
        worker.join(timeout=1.0)


def _pull_params_from_network(*, host: str, port: int, last_optimizer_updates: int, retry_seconds: float):
    sock = connect_with_retry(host, port, retry_seconds)
    try:
        _send_frame(
            sock,
            json.dumps(
                {
                    "last_optimizer_updates": int(last_optimizer_updates),
                },
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        header = json.loads(_recv_frame(sock).decode("utf-8"))
        status = str(header.get("status", ""))
        cur_updates = int(header.get("optimizer_updates", -1))
        if status == "unchanged":
            return None, max(last_optimizer_updates, cur_updates)
        if status != "update":
            raise RuntimeError(f"unknown param sync status: {status}")
        blob = _recv_frame(sock)
        payload = pickle.loads(blob)
        if payload.get("schema") != "gooomoku-param-sync-v1":
            raise RuntimeError(f"unknown param sync payload schema: {payload.get('schema')}")
        pulled_updates = int(payload.get("optimizer_updates", -1))
        pulled_params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
        return pulled_params, pulled_updates
    finally:
        sock.close()


def _save_model_checkpoint(path: str, params, config: dict[str, Any]) -> None:
    payload = {"params": jax.device_get(params), "config": config}
    _write_pickle(path, payload)


def _save_training_checkpoint(
    path: str,
    *,
    params,
    opt_state,
    config: dict[str, Any],
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
    _write_pickle(path, payload)


def _load_checkpoint_payload(path: str):
    return _read_pickle(path)


def _extract_host_tree(params, use_pmap: bool):
    return jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params


def _clone_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def _add_batch_dim_state(state: env.GomokuState) -> env.GomokuState:
    return jax.tree_util.tree_map(lambda x: x[None, ...], state)


def _checkpoint_config(args, optimizer_updates: int, best_step: int) -> dict[str, Any]:
    return {
        "board_size": args.board_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "num_simulations": args.num_simulations,
        "max_num_considered_actions": args.max_num_considered_actions,
        "updates_per_step": args.updates_per_step,
        "temperature": args.temperature,
        "c_lcb": args.c_lcb,
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
        "async_selfplay": args.async_selfplay,
        "cross_process_selfplay": args.cross_process_selfplay,
        "selfplay_actors": args.selfplay_actors,
        "actor_param_sync_updates": args.actor_param_sync_updates,
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


def _apply_dihedral_symmetry_obs_jax(obs: jnp.ndarray, symmetry_id: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.switch(
        symmetry_id,
        (
            lambda x: x,
            lambda x: jnp.rot90(x, k=1, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=2, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=3, axes=(0, 1)),
            lambda x: jnp.flip(x, axis=1),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=1, axes=(0, 1)),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=2, axes=(0, 1)),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=3, axes=(0, 1)),
        ),
        obs,
    )


def _apply_dihedral_symmetry_policy_jax(policy_grid: jnp.ndarray, symmetry_id: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.switch(
        symmetry_id,
        (
            lambda x: x,
            lambda x: jnp.rot90(x, k=1, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=2, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=3, axes=(0, 1)),
            lambda x: jnp.flip(x, axis=1),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=1, axes=(0, 1)),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=2, axes=(0, 1)),
            lambda x: jnp.rot90(jnp.flip(x, axis=1), k=3, axes=(0, 1)),
        ),
        policy_grid,
    )


@jax.jit
def _augment_batch_with_random_symmetry_jax(
    obs: jnp.ndarray,
    policy: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = obs.shape[0]
    board_size = obs.shape[1]
    symmetry_ids = jax.random.randint(rng_key, (batch_size,), minval=0, maxval=8, dtype=jnp.int32)
    policy_grid = policy.reshape((batch_size, board_size, board_size))
    augmented_obs = jax.vmap(_apply_dihedral_symmetry_obs_jax)(obs, symmetry_ids)
    augmented_policy_grid = jax.vmap(_apply_dihedral_symmetry_policy_jax)(policy_grid, symmetry_ids)
    return augmented_obs, augmented_policy_grid.reshape((batch_size, board_size * board_size))


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


def _init_device_replay(
    replay_obs: np.ndarray,
    replay_policy: np.ndarray,
    replay_value: np.ndarray,
    *,
    replay_size: int,
    board_size: int,
):
    num_actions = board_size * board_size
    obs_dev = jnp.zeros((replay_size, board_size, board_size, 4), dtype=jnp.uint8)
    policy_dev = jnp.zeros((replay_size, num_actions), dtype=jnp.float32)
    value_dev = jnp.zeros((replay_size,), dtype=jnp.float32)
    replay_count = int(min(replay_obs.shape[0], replay_size))
    replay_head = replay_count % replay_size if replay_size > 0 else 0
    if replay_count > 0:
        obs_dev = obs_dev.at[:replay_count].set(jnp.asarray(replay_obs[-replay_count:], dtype=jnp.uint8))
        policy_dev = policy_dev.at[:replay_count].set(jnp.asarray(replay_policy[-replay_count:], dtype=jnp.float32))
        value_dev = value_dev.at[:replay_count].set(jnp.asarray(replay_value[-replay_count:], dtype=jnp.float32))
    return obs_dev, policy_dev, value_dev, replay_head, replay_count


def _append_replay_device(
    replay_obs_dev: jnp.ndarray,
    replay_policy_dev: jnp.ndarray,
    replay_value_dev: jnp.ndarray,
    *,
    replay_head: int,
    replay_count: int,
    new_obs: np.ndarray,
    new_policy: np.ndarray,
    new_value: np.ndarray,
):
    capacity = int(replay_obs_dev.shape[0])
    new_count = int(new_obs.shape[0])
    if new_count <= 0 or capacity <= 0:
        return replay_obs_dev, replay_policy_dev, replay_value_dev, replay_head, replay_count

    if new_count >= capacity:
        tail_obs = jnp.asarray(new_obs[-capacity:], dtype=jnp.uint8)
        tail_policy = jnp.asarray(new_policy[-capacity:], dtype=jnp.float32)
        tail_value = jnp.asarray(new_value[-capacity:], dtype=jnp.float32)
        return tail_obs, tail_policy, tail_value, 0, capacity

    first = min(new_count, capacity - replay_head)
    if first > 0:
        replay_obs_dev = replay_obs_dev.at[replay_head : replay_head + first].set(jnp.asarray(new_obs[:first], dtype=jnp.uint8))
        replay_policy_dev = replay_policy_dev.at[replay_head : replay_head + first].set(
            jnp.asarray(new_policy[:first], dtype=jnp.float32)
        )
        replay_value_dev = replay_value_dev.at[replay_head : replay_head + first].set(
            jnp.asarray(new_value[:first], dtype=jnp.float32)
        )

    remaining = new_count - first
    if remaining > 0:
        replay_obs_dev = replay_obs_dev.at[:remaining].set(jnp.asarray(new_obs[first:], dtype=jnp.uint8))
        replay_policy_dev = replay_policy_dev.at[:remaining].set(jnp.asarray(new_policy[first:], dtype=jnp.float32))
        replay_value_dev = replay_value_dev.at[:remaining].set(jnp.asarray(new_value[first:], dtype=jnp.float32))

    replay_head = (replay_head + new_count) % capacity
    replay_count = min(capacity, replay_count + new_count)
    return replay_obs_dev, replay_policy_dev, replay_value_dev, replay_head, replay_count


def _materialize_replay_from_device(
    replay_obs_dev: jnp.ndarray,
    replay_policy_dev: jnp.ndarray,
    replay_value_dev: jnp.ndarray,
    *,
    replay_count: int,
):
    if replay_count <= 0:
        obs_shape = replay_obs_dev.shape[1:]
        policy_dim = replay_policy_dev.shape[1]
        return (
            np.zeros((0,) + obs_shape, dtype=np.uint8),
            np.zeros((0, policy_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    obs_np = np.asarray(jax.device_get(replay_obs_dev[:replay_count]), dtype=np.uint8)
    policy_np = np.asarray(jax.device_get(replay_policy_dev[:replay_count]), dtype=np.float32)
    value_np = np.asarray(jax.device_get(replay_value_dev[:replay_count]), dtype=np.float32)
    return obs_np, policy_np, value_np


def _stabilize_replay_payload_examples(
    flat_obs: np.ndarray,
    flat_policy: np.ndarray,
    flat_value: np.ndarray,
    valid: np.ndarray,
    *,
    fixed_examples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    compact_obs = flat_obs[valid]
    compact_policy = flat_policy[valid]
    compact_value = flat_value[valid]
    compact_count = int(compact_obs.shape[0])

    if fixed_examples <= 0 or compact_count <= 0:
        return (
            np.asarray(compact_obs, dtype=np.uint8),
            np.asarray(compact_policy, dtype=np.float32),
            np.asarray(compact_value, dtype=np.float32),
            compact_count,
        )

    if compact_count >= fixed_examples:
        return (
            np.asarray(compact_obs[:fixed_examples], dtype=np.uint8),
            np.asarray(compact_policy[:fixed_examples], dtype=np.float32),
            np.asarray(compact_value[:fixed_examples], dtype=np.float32),
            fixed_examples,
        )

    pad_index = np.arange(fixed_examples, dtype=np.int32) % compact_count
    return (
        np.asarray(compact_obs[pad_index], dtype=np.uint8),
        np.asarray(compact_policy[pad_index], dtype=np.float32),
        np.asarray(compact_value[pad_index], dtype=np.float32),
        fixed_examples,
    )


def _pack_collect_payload(
    *,
    obs,
    policy,
    value,
    mask,
    winners,
    board_size: int,
    fixed_examples: int,
) -> SelfPlayBatch:
    obs_np, policy_np, value_np, mask_np, winners_np = jax.device_get((obs, policy, value, mask, winners))
    obs_np = np.asarray(obs_np)
    policy_np = np.asarray(policy_np)
    value_np = np.asarray(value_np)
    mask_np = np.asarray(mask_np)
    winners_np = np.asarray(winners_np)

    flat_obs = obs_np.reshape((-1, board_size, board_size, 4))
    flat_policy = policy_np.reshape((-1, board_size * board_size))
    flat_value = value_np.reshape((-1,))
    valid = mask_np.reshape((-1,)).astype(bool)

    packed_obs, packed_policy, packed_value, stored_examples = _stabilize_replay_payload_examples(
        flat_obs,
        flat_policy,
        flat_value,
        valid,
        fixed_examples=fixed_examples,
    )

    return (
        packed_obs,
        packed_policy,
        packed_value,
        int((winners_np == 1).sum()),
        int((winners_np == -1).sum()),
        int((winners_np == 0).sum()),
        stored_examples,
    )


def _maybe_reload_actor_params(
    *,
    resume_from: str | None,
    use_pmap: bool,
    local_devices: int,
    last_optimizer_updates: int,
):
    if resume_from is None:
        return None, last_optimizer_updates
    payload = _load_checkpoint_payload(resume_from)
    if "params" not in payload:
        raise ValueError(f"invalid actor resume checkpoint (missing params): {resume_from}")
    ckpt_optimizer_updates = int(payload.get("optimizer_updates", -1))
    if ckpt_optimizer_updates >= 0 and ckpt_optimizer_updates <= last_optimizer_updates:
        return None, last_optimizer_updates
    loaded = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    if use_pmap:
        loaded = _replicate_tree_for_pmap(loaded, local_devices=local_devices)
    return loaded, ckpt_optimizer_updates


def _run_actor_role(
    *,
    args,
    model: PolicyValueNet,
    params,
    use_pmap: bool,
    local_devices: int,
    process_index: int,
    selfplay_batch_games: int,
):
    endpoint_host, endpoint_port = _resolve_actor_replay_endpoint(args)
    param_endpoint_host = endpoint_host
    param_endpoint_port = int(args.param_sync_port)
    actor_param_source = str(args.actor_param_source).strip().lower()
    use_network_sync = actor_param_source in {"auto", "network"}
    use_checkpoint_sync = actor_param_source in {"auto", "checkpoint"} and (args.resume_from is not None)

    games_per_device = selfplay_batch_games // local_devices if use_pmap else selfplay_batch_games
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
        c_lcb=args.c_lcb,
    )
    collect_step = make_pmap_collect_step(play_many_games_fn) if use_pmap else play_many_games_fn

    if use_pmap:
        params = _replicate_tree_for_pmap(params, local_devices=local_devices)

    last_optimizer_updates = -1
    if use_network_sync:
        try:
            net_params, pulled_updates = _pull_params_from_network(
                host=param_endpoint_host,
                port=param_endpoint_port,
                last_optimizer_updates=last_optimizer_updates,
                retry_seconds=args.replay_connect_retry_seconds,
            )
            if net_params is not None:
                params = _replicate_tree_for_pmap(net_params, local_devices=local_devices) if use_pmap else net_params
                last_optimizer_updates = pulled_updates
                print(
                    f"actor[{process_index}] loaded params from network {param_endpoint_host}:{param_endpoint_port} "
                    f"optimizer_updates={last_optimizer_updates}"
                )
        except BaseException as exc:
            if actor_param_source == "network":
                raise RuntimeError(f"actor network param sync init failed: {exc}") from exc
            print(f"actor[{process_index}] network param sync unavailable, fallback to checkpoint: {exc}")

    if last_optimizer_updates < 0 and use_checkpoint_sync:
        loaded_params, last_optimizer_updates = _maybe_reload_actor_params(
            resume_from=args.resume_from,
            use_pmap=use_pmap,
            local_devices=local_devices,
            last_optimizer_updates=last_optimizer_updates,
        )
        if loaded_params is not None:
            params = loaded_params
            print(f"actor[{process_index}] loaded params from {args.resume_from} optimizer_updates={last_optimizer_updates}")

    sock = connect_with_retry(endpoint_host, endpoint_port, args.replay_connect_retry_seconds)
    print(
        f"actor[{process_index}] connected to learner replay endpoint {endpoint_host}:{endpoint_port} "
        f"use_pmap={use_pmap} local_device_count={local_devices} games_per_batch={selfplay_batch_games}"
    )

    rng_key = jax.random.PRNGKey(args.seed + 300001 * (process_index + 1))
    sent_batches = 0
    sent_examples = 0
    steps_limit = int(args.actor_steps)
    sync_every = int(args.actor_sync_every_batches)
    start_time = time.perf_counter()
    try:
        while steps_limit <= 0 or sent_batches < steps_limit:
            next_batch_idx = sent_batches + 1
            print(f"actor[{process_index}] collecting batch={next_batch_idx}")
            rng_key, collect_key = jax.random.split(rng_key)
            collect_start = time.perf_counter()
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
            print(f"actor[{process_index}] collect_done batch={next_batch_idx}")

            print(f"actor[{process_index}] payload_start batch={next_batch_idx}")
            payload_start = time.perf_counter()
            payload = _pack_collect_payload(
                obs=obs,
                policy=policy,
                value=value,
                mask=mask,
                winners=winners,
                board_size=args.board_size,
                fixed_examples=args.replay_fixed_update_size,
            )
            payload_ms = (time.perf_counter() - payload_start) * 1000.0
            print(
                f"actor[{process_index}] payload_done batch={next_batch_idx} "
                f"examples={payload[6]} payload_ms={payload_ms:.1f}"
            )
            collect_ms = (time.perf_counter() - collect_start) * 1000.0
            print(f"actor[{process_index}] upload_start batch={next_batch_idx}")
            upload_start = time.perf_counter()
            send_selfplay_batch(sock, payload)
            upload_ms = (time.perf_counter() - upload_start) * 1000.0
            print(
                f"actor[{process_index}] upload_done batch={next_batch_idx} "
                f"upload_ms={upload_ms:.1f}"
            )
            sent_batches += 1
            sent_examples += payload[6]

            if sync_every > 0 and (sent_batches % sync_every == 0):
                synced = False
                if use_network_sync:
                    try:
                        net_params, pulled_updates = _pull_params_from_network(
                            host=param_endpoint_host,
                            port=param_endpoint_port,
                            last_optimizer_updates=last_optimizer_updates,
                            retry_seconds=args.replay_connect_retry_seconds,
                        )
                        if net_params is not None:
                            params = _replicate_tree_for_pmap(net_params, local_devices=local_devices) if use_pmap else net_params
                            last_optimizer_updates = pulled_updates
                            synced = True
                            print(
                                f"actor[{process_index}] synced params from network {param_endpoint_host}:{param_endpoint_port} "
                                f"optimizer_updates={last_optimizer_updates}"
                            )
                    except BaseException as exc:
                        if actor_param_source == "network":
                            raise RuntimeError(f"actor network param sync failed: {exc}") from exc
                        print(f"actor[{process_index}] network param sync skipped: {exc}")

                if (not synced) and use_checkpoint_sync:
                    loaded_params, last_optimizer_updates = _maybe_reload_actor_params(
                        resume_from=args.resume_from,
                        use_pmap=use_pmap,
                        local_devices=local_devices,
                        last_optimizer_updates=last_optimizer_updates,
                    )
                    if loaded_params is not None:
                        params = loaded_params
                        print(
                            f"actor[{process_index}] synced params from {args.resume_from} "
                            f"optimizer_updates={last_optimizer_updates}"
                        )

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            print(
                f"actor[{process_index}] sent_batches={sent_batches} sent_examples={sent_examples} "
                f"collect_ms={collect_ms:.1f} "
                f"examples_per_sec={sent_examples / elapsed:.1f}"
            )
    finally:
        sock.close()


def _start_remote_replay_listener(host: str, port: int, queue_size: int):
    replay_queue: queue_lib.Queue[SelfPlayBatch] = queue_lib.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    listener_errors: list[BaseException] = []
    client_threads: list[threading.Thread] = []
    client_threads_lock = threading.Lock()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    server_sock.settimeout(1.0)

    def client_loop(conn: socket.socket, addr: tuple[str, int]) -> None:
        with conn:
            while not stop_event.is_set():
                try:
                    payload = recv_selfplay_batch(conn)
                except ReplayWireError as exc:
                    if "socket closed while receiving replay frame" in str(exc):
                        # Peer closed connection gracefully between frames.
                        break
                    if not stop_event.is_set():
                        listener_errors.append(RuntimeError(f"replay wire error from {addr}: {exc}"))
                    break
                except OSError as exc:
                    if exc.errno in (32, 104):
                        # Broken pipe / connection reset from actor shutdown.
                        break
                    if not stop_event.is_set():
                        listener_errors.append(RuntimeError(f"socket error from {addr}: {exc}"))
                    break
                while not stop_event.is_set():
                    try:
                        replay_queue.put(payload, timeout=0.2)
                        break
                    except queue_lib.Full:
                        continue

    def accept_loop() -> None:
        while not stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not stop_event.is_set():
                    listener_errors.append(RuntimeError(f"replay listener accept failed: {exc}"))
                break
            worker = threading.Thread(target=client_loop, args=(conn, addr), daemon=True, name=f"replay-client-{addr[0]}:{addr[1]}")
            worker.start()
            with client_threads_lock:
                client_threads.append(worker)

    accept_thread = threading.Thread(target=accept_loop, daemon=True, name="replay-listener")
    accept_thread.start()
    return replay_queue, stop_event, listener_errors, server_sock, accept_thread, client_threads, client_threads_lock


def _stop_remote_replay_listener(
    *,
    stop_event,
    server_sock: socket.socket | None,
    accept_thread: threading.Thread | None,
    client_threads: list[threading.Thread],
    client_threads_lock: threading.Lock,
) -> None:
    stop_event.set()
    if server_sock is not None:
        server_sock.close()
    if accept_thread is not None:
        accept_thread.join(timeout=5.0)
    with client_threads_lock:
        snapshot = list(client_threads)
    for worker in snapshot:
        worker.join(timeout=1.0)


def main() -> None:
    configure_jax_runtime(app_name="train", repo_root=REPO_ROOT)
    parser = argparse.ArgumentParser(description="Minimal JAX+mctx gomoku trainer.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--role", choices=("all", "learner", "actor"), default="all")
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
    parser.add_argument("--c-lcb", type=float, default=0.0)
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
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/latest.pkl")
    parser.add_argument("--disable-pmap", action="store_true")
    parser.add_argument("--distributed-init", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--jax-platforms",
        type=str,
        default="",
        help="Optional JAX_PLATFORMS for learner process (e.g. 'tpu', 'cpu', 'gpu', '').",
    )
    parser.add_argument("--async-selfplay", action="store_true")
    parser.add_argument("--cross-process-selfplay", action="store_true")
    parser.add_argument(
        "--actor-jax-platforms",
        type=str,
        default="cpu",
        help="JAX_PLATFORMS for cross-process selfplay actors (e.g. 'cpu', 'tpu', 'gpu').",
    )
    parser.add_argument("--async-selfplay-queue-size", type=int, default=4)
    parser.add_argument("--selfplay-actors", type=int, default=1)
    parser.add_argument("--actor-param-sync-updates", type=int, default=1)
    parser.add_argument(
        "--replay-fixed-update-size",
        type=int,
        default=0,
        help=(
            "If > 0, force each self-play payload to a fixed example count "
            "(truncate/repeat valid rows) to stabilize replay append shapes and reduce recompiles."
        ),
    )
    parser.add_argument("--selfplay-batch-games", type=int, default=None)
    parser.add_argument("--replay-host", type=str, default="auto")
    parser.add_argument("--replay-port", type=int, default=19091)
    parser.add_argument("--param-sync-port", type=int, default=19092)
    parser.add_argument("--remote-replay-queue-size", type=int, default=64)
    parser.add_argument("--replay-connect-retry-seconds", type=float, default=2.0)
    parser.add_argument("--actor-param-source", choices=("auto", "network", "checkpoint"), default="auto")
    parser.add_argument("--param-sync-every-steps", type=int, default=1)
    parser.add_argument("--replay-endpoint-uri", type=str, default="")
    parser.add_argument("--replay-advertise-host", type=str, default="auto")
    parser.add_argument("--replay-endpoint-wait-seconds", type=float, default=180.0)
    parser.add_argument("--actor-steps", type=int, default=0)
    parser.add_argument("--actor-sync-every-batches", type=int, default=8)
    parser.add_argument("--wait-log-interval-seconds", type=float, default=30.0)
    parser.add_argument("--phase-log-threshold-ms", type=float, default=5000.0)
    parser.add_argument("--detailed-step-log", action="store_true")
    parser.add_argument(
        "--disable-fused-train-updates",
        action="store_true",
        help="Disable fused multi-update train step and keep one-jit-call-per-update behavior.",
    )
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
    if args.async_selfplay_queue_size < 1:
        raise ValueError("async-selfplay-queue-size must be >= 1")
    if args.selfplay_actors < 1:
        raise ValueError("selfplay-actors must be >= 1")
    if args.actor_param_sync_updates < 1:
        raise ValueError("actor-param-sync-updates must be >= 1")
    if args.replay_fixed_update_size < 0:
        raise ValueError("replay-fixed-update-size must be >= 0")
    if args.remote_replay_queue_size < 1:
        raise ValueError("remote-replay-queue-size must be >= 1")
    if args.replay_port <= 0:
        raise ValueError("replay-port must be > 0")
    if args.param_sync_port <= 0:
        raise ValueError("param-sync-port must be > 0")
    if args.replay_connect_retry_seconds <= 0:
        raise ValueError("replay-connect-retry-seconds must be > 0")
    if args.param_sync_every_steps < 1:
        raise ValueError("param-sync-every-steps must be >= 1")
    if args.replay_endpoint_wait_seconds <= 0:
        raise ValueError("replay-endpoint-wait-seconds must be > 0")
    if args.actor_steps < 0:
        raise ValueError("actor-steps must be >= 0")
    if args.actor_sync_every_batches < 0:
        raise ValueError("actor-sync-every-batches must be >= 0")
    if args.wait_log_interval_seconds <= 0:
        raise ValueError("wait-log-interval-seconds must be > 0")
    if args.phase_log_threshold_ms < 0:
        raise ValueError("phase-log-threshold-ms must be >= 0")
    if args.role == "actor" and args.actor_param_source == "checkpoint" and args.resume_from is None:
        raise ValueError("--actor-param-source=checkpoint requires --resume-from")

    if args.role in {"actor", "learner"}:
        args.async_selfplay = False
        args.cross_process_selfplay = False
    actor_jax_platforms = args.actor_jax_platforms.strip()
    learner_jax_platforms = args.jax_platforms.strip()
    if learner_jax_platforms:
        os.environ["JAX_PLATFORMS"] = learner_jax_platforms
        os.environ.pop("JAX_PLATFORM_NAME", None)
        jax.config.update("jax_platforms", learner_jax_platforms)
    if args.cross_process_selfplay and "tpu" in {p.strip() for p in actor_jax_platforms.split(",") if p.strip()}:
        raise ValueError(
            "cross-process selfplay actors cannot use TPU in this process model (libtpu is process-exclusive). "
            "Use --actor-jax-platforms=cpu with --cross-process-selfplay; "
            "if you need TPU selfplay, disable --cross-process-selfplay and run in-process async selfplay."
        )

    distributed_initialized = False
    if args.distributed_init != "off":
        should_try_distributed = args.distributed_init == "on" or any(
            k in os.environ for k in ("TPU_WORKER_ID", "CLOUD_TPU_TASK_ID", "MEGASCALE_COORDINATOR_ADDRESS")
        )
        if should_try_distributed:
            try:
                jax.distributed.initialize()
                distributed_initialized = True
            except RuntimeError as exc:
                if _looks_like_tpu_runtime_error(exc):
                    _print_tpu_startup_diagnostics(stage="jax.distributed.initialize()", exc=exc)
                    raise
                if args.distributed_init == "on":
                    raise
                print(f"distributed-init skipped: {exc}")

    try:
        _ = jax.devices()
    except RuntimeError as exc:
        if _looks_like_tpu_runtime_error(exc):
            _print_tpu_startup_diagnostics(stage="backend preflight", exc=exc)
            raise
        raise

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
    replay_obs = np.zeros((0, args.board_size, args.board_size, 4), dtype=np.uint8)
    replay_policy = np.zeros((0, args.board_size * args.board_size), dtype=np.float32)
    replay_value = np.zeros((0,), dtype=np.float32)
    np_rng = np.random.default_rng(args.seed + 13)
    optimizer_updates = 0
    best_params = _clone_tree(params)
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
            replay_obs = np.asarray(payload["replay_obs"], dtype=np.uint8)
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
            best_params = _clone_tree(params)
        best_step = int(payload.get("best_step", 0))

        print(
            f"resumed from {args.resume_from}: last_step={last_step} start_step={start_step} "
            f"optimizer_updates={optimizer_updates} replay={replay_obs.shape[0]}"
        )

    process_count = jax.process_count()
    process_index = jax.process_index()
    is_chief = process_index == 0


    local_devices = jax.local_device_count()
    global_devices = jax.device_count()
    use_pmap = (not args.disable_pmap) and local_devices > 1
    if args.role != "actor" and use_pmap and (args.batch_size % local_devices != 0):
        raise ValueError(f"batch-size must be divisible by local_device_count={local_devices}")
    selfplay_batch_games = args.selfplay_batch_games or args.games_per_step
    if selfplay_batch_games < 1:
        raise ValueError("selfplay-batch-games must be >= 1")
    if use_pmap and (selfplay_batch_games % local_devices != 0):
        raise ValueError(f"games-per-step must be divisible by local_device_count={local_devices}")
    if args.role != "actor" and use_pmap and args.arena_every_steps > 0 and (args.arena_games % local_devices != 0):
        raise ValueError(f"arena-games must be divisible by local_device_count={local_devices} when pmap is enabled")

    if args.role == "actor":
        _run_actor_role(
            args=args,
            model=model,
            params=params,
            use_pmap=use_pmap,
            local_devices=local_devices,
            process_index=process_index,
            selfplay_batch_games=selfplay_batch_games,
        )
        return

    games_per_device = selfplay_batch_games // local_devices if use_pmap else selfplay_batch_games
    fused_train_updates = args.updates_per_step > 1 and (not args.disable_fused_train_updates)
    actor_device_groups = None
    actor_devices_per_group = 0
    if args.async_selfplay and (not args.cross_process_selfplay) and use_pmap:
        if args.selfplay_actors > local_devices:
            raise ValueError(
                f"selfplay-actors={args.selfplay_actors} cannot exceed local_device_count={local_devices} in in-process TPU mode"
            )
        if local_devices % args.selfplay_actors != 0:
            raise ValueError(
                f"local_device_count={local_devices} must be divisible by selfplay-actors={args.selfplay_actors} "
                "for in-process TPU device partitioning"
            )
        actor_devices_per_group = local_devices // args.selfplay_actors
        local_device_list = list(jax.local_devices())
        actor_device_groups = [
            tuple(local_device_list[i * actor_devices_per_group : (i + 1) * actor_devices_per_group])
            for i in range(args.selfplay_actors)
        ]
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
        c_lcb=args.c_lcb,
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
        train_step_multi = make_pmap_multi_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = make_pmap_collect_step(play_many_games_fn)
        params = _replicate_tree_for_pmap(params, local_devices=local_devices)
        opt_state = _replicate_tree_for_pmap(opt_state, local_devices=local_devices)
        best_params_repl = _replicate_tree_for_pmap(best_params, local_devices=local_devices)
        print(
            f"pmap enabled: process={process_index}/{process_count}, local_device_count={local_devices}, "
            f"global_device_count={global_devices}, distributed_initialized={distributed_initialized}, "
            f"per_device_batch={per_device_batch}, "
            f"selfplay_games_per_device={games_per_device}, async_selfplay={args.async_selfplay}, "
            f"cross_process_selfplay={args.cross_process_selfplay}, "
            f"selfplay_actors={args.selfplay_actors}, actor_param_sync_updates={args.actor_param_sync_updates}, "
            f"updates_per_step={args.updates_per_step}, "
            f"fused_train_updates={fused_train_updates}, "
            f"replay_fixed_update_size={args.replay_fixed_update_size}, "
            f"compute_dtype={args.compute_dtype}, param_dtype={args.param_dtype}"
        )
    else:
        train_step = make_single_train_step(model, optimizer, weight_decay=args.weight_decay)
        train_step_multi = make_single_multi_train_step(model, optimizer, weight_decay=args.weight_decay)
        collect_step = play_many_games_fn
        per_device_batch = args.batch_size
        best_params_repl = None
        print(
            f"single-device mode: process={process_index}/{process_count}, local_device_count={local_devices}, "
            f"global_device_count={global_devices}, distributed_initialized={distributed_initialized}, "
            f"updates_per_step={args.updates_per_step}, "
            f"async_selfplay={args.async_selfplay}, cross_process_selfplay={args.cross_process_selfplay}, "
            f"selfplay_actors={args.selfplay_actors}, "
            f"actor_param_sync_updates={args.actor_param_sync_updates}, "
            f"fused_train_updates={fused_train_updates}, "
            f"replay_fixed_update_size={args.replay_fixed_update_size}, "
            f"compute_dtype={args.compute_dtype}, param_dtype={args.param_dtype}"
        )

    best_params = jax.tree_util.tree_map(jnp.asarray, best_params)
    replay_obs_dev, replay_policy_dev, replay_value_dev, replay_head, replay_count = _init_device_replay(
        replay_obs,
        replay_policy,
        replay_value,
        replay_size=args.replay_size,
        board_size=args.board_size,
    )
    best_path = _best_checkpoint_path(args.output)

    def maybe_save_training_checkpoint(step: int, reason: str) -> float:
        if not is_chief:
            return 0.0
        if args.checkpoint_every_steps <= 0 or (step % args.checkpoint_every_steps != 0):
            return 0.0
        save_start = time.perf_counter()
        current_params = _extract_host_tree(params, use_pmap=use_pmap)
        current_opt_state = _extract_host_tree(opt_state, use_pmap=use_pmap)
        replay_obs_np, replay_policy_np, replay_value_np = _materialize_replay_from_device(
            replay_obs_dev,
            replay_policy_dev,
            replay_value_dev,
            replay_count=replay_count,
        )
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
            replay_obs=replay_obs_np,
            replay_policy=replay_policy_np,
            replay_value=replay_value_np,
            best_params=best_params,
            best_step=best_step,
        )
        checkpoint_ms = (time.perf_counter() - save_start) * 1000.0
        print(f"saved training checkpoint ({reason}) to {args.output} checkpoint_ms={checkpoint_ms:.1f}")
        return checkpoint_ms

    completed_step = start_step - 1
    replay_queue: queue_lib.Queue[SelfPlayBatch] | None = None
    actor_stop: Any = None
    actor_errors: list[ActorError] = []
    params_ref_lock: threading.Lock | None = None
    params_ref: list[Any] | None = None
    actor_threads: list[threading.Thread] = []
    actor_processes: list[Any] = []
    actor_params_queues: list[Any] = []
    actor_output_queue: Any = None
    actor_stats_lock = threading.Lock()
    actor_batches_total = 0
    actor_examples_total = 0
    learner_wait_total_sec = 0.0
    learner_wait_count = 0
    train_start_time = time.perf_counter()
    remote_stop_event: threading.Event | None = None
    remote_listener_errors: list[BaseException] = []
    remote_listener_error_count = 0
    remote_server_sock: socket.socket | None = None
    remote_accept_thread: threading.Thread | None = None
    remote_client_threads: list[threading.Thread] = []
    remote_client_threads_lock: threading.Lock | None = None
    param_sync_snapshot_lock: threading.Lock | None = None
    param_sync_snapshot_state: dict[str, Any] | None = None
    param_sync_stop_event: threading.Event | None = None
    param_sync_errors: list[BaseException] = []
    param_sync_error_count = 0
    param_sync_server_sock: socket.socket | None = None
    param_sync_accept_thread: threading.Thread | None = None
    param_sync_client_threads: list[threading.Thread] = []
    param_sync_client_threads_lock: threading.Lock | None = None
    wait_log_interval_sec = args.wait_log_interval_seconds
    learner_last_wait_log_at = time.perf_counter()
    async_last_wait_log_at = learner_last_wait_log_at

    def collect_new_examples(params_for_collect, collect_rng_key):
        if use_pmap:
            collect_keys = jax.random.split(collect_rng_key, local_devices)
            collect_temperature = jnp.full((local_devices,), jnp.float32(args.temperature), dtype=jnp.float32)
            obs, policy, value, mask, _, winners = collect_step(
                params_for_collect,
                collect_keys,
                collect_temperature,
            )
        else:
            obs, policy, value, mask, _, winners = collect_step(
                params_for_collect,
                collect_rng_key,
                jnp.float32(args.temperature),
            )
        return _pack_collect_payload(
            obs=obs,
            policy=policy,
            value=value,
            mask=mask,
            winners=winners,
            board_size=args.board_size,
            fixed_examples=args.replay_fixed_update_size,
        )

    if args.role == "learner":
        bind_host = args.replay_host.strip()
        if (not bind_host) or bind_host.lower() == "auto":
            bind_host = "0.0.0.0"
        advertise_host = args.replay_advertise_host.strip()
        if (not advertise_host) or advertise_host.lower() == "auto":
            if bind_host not in {"0.0.0.0", "::"}:
                advertise_host = bind_host
            else:
                advertise_host = _detect_advertise_host()
        if advertise_host in {"0.0.0.0", "::"}:
            advertise_host = _detect_advertise_host()
        endpoint_uri = _derive_replay_endpoint_uri(
            role="learner",
            replay_endpoint_uri=args.replay_endpoint_uri,
            output=args.output,
            resume_from=args.resume_from,
        )
        (
            replay_queue,
            remote_stop_event,
            remote_listener_errors,
            remote_server_sock,
            remote_accept_thread,
            remote_client_threads,
            remote_client_threads_lock,
        ) = _start_remote_replay_listener(
            host=bind_host,
            port=args.replay_port,
            queue_size=args.remote_replay_queue_size,
        )
        param_sync_snapshot_lock = threading.Lock()
        param_sync_snapshot_state = {
            "optimizer_updates": int(optimizer_updates),
            "blob": _build_param_sync_blob(
                params=params,
                use_pmap=use_pmap,
                optimizer_updates=optimizer_updates,
            ),
        }
        (
            param_sync_stop_event,
            param_sync_errors,
            param_sync_server_sock,
            param_sync_accept_thread,
            param_sync_client_threads,
            param_sync_client_threads_lock,
        ) = _start_param_sync_server(
            host=bind_host,
            port=args.param_sync_port,
            snapshot_state=param_sync_snapshot_state,
            snapshot_lock=param_sync_snapshot_lock,
        )
        if endpoint_uri is not None:
            _publish_replay_endpoint(endpoint_uri, host=advertise_host, port=args.replay_port)
        if is_chief:
            print(
                f"learner replay listener ready bind={bind_host}:{args.replay_port} "
                f"advertise={advertise_host}:{args.replay_port} queue_size={args.remote_replay_queue_size} "
                f"endpoint_uri={endpoint_uri} param_sync_port={args.param_sync_port}"
            )
    elif args.async_selfplay:
        replay_queue = queue_lib.Queue(maxsize=args.async_selfplay_queue_size)
        if args.cross_process_selfplay:
            ctx = mp.get_context("spawn")
            actor_stop = ctx.Event()
            actor_output_queue = ctx.Queue(maxsize=args.async_selfplay_queue_size)
            params_seed = _extract_host_tree(params, use_pmap=use_pmap)
            params_seed = jax.device_get(params_seed)
            proc_cfg = {
                "board_size": args.board_size,
                "channels": args.channels,
                "blocks": args.blocks,
                "compute_dtype": args.compute_dtype,
                "param_dtype": args.param_dtype,
                "num_simulations": args.num_simulations,
                "max_num_considered_actions": args.max_num_considered_actions,
                "num_games": selfplay_batch_games,
                "temperature_drop_move": args.temperature_drop_move,
                "final_temperature": args.final_temperature,
                "root_dirichlet_fraction": args.root_dirichlet_fraction,
                "root_dirichlet_alpha": args.root_dirichlet_alpha,
                "c_lcb": args.c_lcb,
                "temperature": args.temperature,
                "seed": args.seed,
                "actor_jax_platforms": actor_jax_platforms,
                "replay_fixed_update_size": args.replay_fixed_update_size,
            }
            # Spawned actors inherit parent env. Set actor JAX platform in parent
            # before Process.start() so child bootstrap sees the intended backend.
            prev_jax_platforms = os.environ.get("JAX_PLATFORMS")
            prev_jax_platform_name = os.environ.get("JAX_PLATFORM_NAME")
            if actor_jax_platforms:
                os.environ["JAX_PLATFORMS"] = actor_jax_platforms
                os.environ.pop("JAX_PLATFORM_NAME", None)
            try:
                for actor_idx in range(args.selfplay_actors):
                    param_q = ctx.Queue(maxsize=1)
                    param_q.put(params_seed)
                    actor_params_queues.append(param_q)
                    actor_proc = ctx.Process(
                        target=_run_cross_process_actor,
                        args=(actor_idx, proc_cfg, param_q, actor_output_queue, actor_stop),
                        name=f"selfplay-proc-{actor_idx}",
                        daemon=True,
                    )
                    actor_proc.start()
                    actor_processes.append(actor_proc)
            finally:
                if actor_jax_platforms:
                    if prev_jax_platforms is None:
                        os.environ.pop("JAX_PLATFORMS", None)
                    else:
                        os.environ["JAX_PLATFORMS"] = prev_jax_platforms
                    if prev_jax_platform_name is None:
                        os.environ.pop("JAX_PLATFORM_NAME", None)
                    else:
                        os.environ["JAX_PLATFORM_NAME"] = prev_jax_platform_name
        else:
            actor_stop = threading.Event()
            params_ref_lock = threading.Lock()
            params_ref = [_clone_tree(params)]
            actor_collect_steps = None
            if use_pmap:
                assert actor_device_groups is not None
                actor_collect_steps = [
                    make_pmap_collect_step(play_many_games_fn, devices=devices)
                    for devices in actor_device_groups
                ]

            def actor_loop(actor_idx: int) -> None:
                nonlocal actor_batches_total, actor_examples_total
                actor_rng = jax.random.PRNGKey(args.seed + 9973 + actor_idx * 100003)
                while actor_stop is not None and (not actor_stop.is_set()):
                    try:
                        if params_ref_lock is None or params_ref is None:
                            break
                        with params_ref_lock:
                            collect_params = params_ref[0]
                        actor_rng, collect_key = jax.random.split(actor_rng)
                        if use_pmap and actor_collect_steps is not None:
                            assert actor_devices_per_group > 0
                            start = actor_idx * actor_devices_per_group
                            end = start + actor_devices_per_group
                            collect_params_actor = jax.tree_util.tree_map(lambda x: x[start:end], collect_params)
                            collect_keys = jax.random.split(collect_key, actor_devices_per_group)
                            collect_temperature = jnp.full(
                                (actor_devices_per_group,),
                                jnp.float32(args.temperature),
                                dtype=jnp.float32,
                            )
                            obs, policy, value, mask, _, winners = actor_collect_steps[actor_idx](
                                collect_params_actor,
                                collect_keys,
                                collect_temperature,
                            )
                            payload = _pack_collect_payload(
                                obs=obs,
                                policy=policy,
                                value=value,
                                mask=mask,
                                winners=winners,
                                board_size=args.board_size,
                                fixed_examples=args.replay_fixed_update_size,
                            )
                        else:
                            payload = collect_new_examples(collect_params, collect_key)
                        assert replay_queue is not None
                        while actor_stop is not None and (not actor_stop.is_set()):
                            try:
                                replay_queue.put(payload, timeout=0.2)
                                with actor_stats_lock:
                                    actor_batches_total += 1
                                    actor_examples_total += payload[6]
                                break
                            except queue_lib.Full:
                                continue
                    except BaseException as exc:
                        actor_errors.append((actor_idx, exc))
                        if actor_stop is not None:
                            actor_stop.set()
                        break

            for actor_idx in range(args.selfplay_actors):
                actor_thread = threading.Thread(
                    target=actor_loop,
                    args=(actor_idx,),
                    name=f"selfplay-actor-{actor_idx}",
                    daemon=True,
                )
                actor_thread.start()
                actor_threads.append(actor_thread)

    if start_step > args.train_steps:
        print(
            f"resume start_step={start_step} is beyond train_steps={args.train_steps}; "
            "skipping update loop and saving checkpoint."
        )
    try:
        for step in range(start_step, args.train_steps + 1):
            completed_step = step
            step_start = time.perf_counter()
            collect_wait_ms = 0.0
            replay_append_ms = 0.0
            train_updates_ms = 0.0
            param_snapshot_ms = 0.0
            arena_ms = 0.0
            checkpoint_ms = 0.0
            if actor_errors:
                actor_idx, exc = actor_errors[0]
                raise RuntimeError(f"async self-play actor[{actor_idx}] failed: {exc}") from exc

            if args.role == "learner":
                assert replay_queue is not None
                wait_start = time.perf_counter()
                while True:
                    if remote_listener_errors:
                        if len(remote_listener_errors) > remote_listener_error_count:
                            latest = remote_listener_errors[-1]
                            remote_listener_error_count = len(remote_listener_errors)
                            print(f"learner replay listener warning: {latest}")
                    if param_sync_errors:
                        if len(param_sync_errors) > param_sync_error_count:
                            latest = param_sync_errors[-1]
                            param_sync_error_count = len(param_sync_errors)
                            print(f"learner param-sync warning: {latest}")
                    try:
                        (
                            new_obs,
                            new_policy,
                            new_value,
                            black_win,
                            white_win,
                            draw,
                            new_examples,
                        ) = replay_queue.get(timeout=0.5)
                        with actor_stats_lock:
                            actor_batches_total += 1
                            actor_examples_total += new_examples
                        break
                    except queue_lib.Empty:
                        now = time.perf_counter()
                        if now - learner_last_wait_log_at >= wait_log_interval_sec:
                            learner_last_wait_log_at = now
                            print(
                                "learner waiting for replay batch "
                                f"qsize={replay_queue.qsize()} "
                                f"listener_warnings={remote_listener_error_count} "
                                f"param_sync_warnings={param_sync_error_count}"
                            )
                        continue
                wait_sec = time.perf_counter() - wait_start
                collect_wait_ms = wait_sec * 1000.0
                learner_wait_total_sec += wait_sec
                learner_wait_count += 1
            elif args.async_selfplay:
                assert actor_stop is not None
                wait_start = time.perf_counter()
                while True:
                    if actor_errors:
                        actor_idx, exc = actor_errors[0]
                        raise RuntimeError(f"async self-play actor[{actor_idx}] failed: {exc}") from exc
                    try:
                        if args.cross_process_selfplay:
                            assert actor_output_queue is not None
                            msg_type, msg_payload = actor_output_queue.get(timeout=0.5)
                            if msg_type == "error":
                                actor_idx, err_text = msg_payload
                                actor_errors.append((int(actor_idx), RuntimeError(err_text)))
                                continue
                            (
                                new_obs,
                                new_policy,
                                new_value,
                                black_win,
                                white_win,
                                draw,
                                new_examples,
                            ) = msg_payload
                            with actor_stats_lock:
                                actor_batches_total += 1
                                actor_examples_total += new_examples
                        else:
                            assert replay_queue is not None
                            (
                                new_obs,
                                new_policy,
                                new_value,
                                black_win,
                                white_win,
                                draw,
                                new_examples,
                            ) = replay_queue.get(timeout=0.5)
                        break
                    except queue_lib.Empty:
                        if actor_stop.is_set() and actor_errors:
                            actor_idx, exc = actor_errors[0]
                            raise RuntimeError(f"async self-play actor[{actor_idx}] failed: {exc}") from exc
                        now = time.perf_counter()
                        if now - async_last_wait_log_at >= wait_log_interval_sec:
                            async_last_wait_log_at = now
                            assert replay_queue is not None
                            print(
                                "async selfplay waiting for batch "
                                f"qsize={replay_queue.qsize()} actor_errors={len(actor_errors)}"
                            )
                        continue
                wait_sec = time.perf_counter() - wait_start
                collect_wait_ms = wait_sec * 1000.0
                learner_wait_total_sec += wait_sec
                learner_wait_count += 1
            else:
                rng_key, collect_key = jax.random.split(rng_key)
                collect_start = time.perf_counter()
                (
                    new_obs,
                    new_policy,
                    new_value,
                    black_win,
                    white_win,
                    draw,
                    new_examples,
                ) = collect_new_examples(params, collect_key)
                collect_wait_ms = (time.perf_counter() - collect_start) * 1000.0

            replay_append_start = time.perf_counter()
            replay_obs_dev, replay_policy_dev, replay_value_dev, replay_head, replay_count = _append_replay_device(
                replay_obs_dev,
                replay_policy_dev,
                replay_value_dev,
                replay_head=replay_head,
                replay_count=replay_count,
                new_obs=new_obs,
                new_policy=new_policy,
                new_value=new_value,
            )
            replay_append_ms = (time.perf_counter() - replay_append_start) * 1000.0

            if replay_count < args.batch_size:
                print(f"step={step} replay={replay_count} waiting for enough samples")
                checkpoint_ms = maybe_save_training_checkpoint(step, "replay-wait")
                continue

            loss_sum = 0.0
            policy_sum = 0.0
            value_sum = 0.0
            train_updates_start = time.perf_counter()
            if fused_train_updates:
                obs_batches = []
                policy_batches = []
                value_batches = []
                for _ in range(args.updates_per_step):
                    rng_key, sample_key = jax.random.split(rng_key)
                    sample_ids = jax.random.randint(
                        sample_key,
                        shape=(args.batch_size,),
                        minval=0,
                        maxval=replay_count,
                        dtype=jnp.int32,
                    )
                    obs = replay_obs_dev[sample_ids]
                    policy_target = replay_policy_dev[sample_ids]
                    value_target = replay_value_dev[sample_ids]
                    if not args.disable_symmetry_augmentation:
                        rng_key, aug_key = jax.random.split(rng_key)
                        obs, policy_target = _augment_batch_with_random_symmetry_jax(obs, policy_target, aug_key)

                    if use_pmap:
                        obs = obs.reshape((local_devices, per_device_batch, args.board_size, args.board_size, 4))
                        policy_target = policy_target.reshape((local_devices, per_device_batch, -1))
                        value_target = value_target.reshape((local_devices, per_device_batch))

                    obs_batches.append(obs)
                    policy_batches.append(policy_target)
                    value_batches.append(value_target)

                obs_stack = jnp.stack(obs_batches, axis=0)
                policy_stack = jnp.stack(policy_batches, axis=0)
                value_stack = jnp.stack(value_batches, axis=0)
                if use_pmap:
                    # pmap maps over leading axis (local_devices). For fused updates we first
                    # stack by update index, so move device axis to front before calling pmap.
                    # Before: [updates_per_step, local_devices, per_device_batch, ...]
                    # After:  [local_devices, updates_per_step, per_device_batch, ...]
                    obs_stack = jnp.swapaxes(obs_stack, 0, 1)
                    policy_stack = jnp.swapaxes(policy_stack, 0, 1)
                    value_stack = jnp.swapaxes(value_stack, 0, 1)
                params, opt_state, loss_mean, policy_mean, value_mean = train_step_multi(
                    params,
                    opt_state,
                    obs_stack,
                    policy_stack,
                    value_stack,
                )
                if use_pmap:
                    loss_scalar = float(loss_mean[0])
                    policy_scalar = float(policy_mean[0])
                    value_scalar = float(value_mean[0])
                else:
                    loss_scalar = float(loss_mean)
                    policy_scalar = float(policy_mean)
                    value_scalar = float(value_mean)
                loss_sum = loss_scalar * args.updates_per_step
                policy_sum = policy_scalar * args.updates_per_step
                value_sum = value_scalar * args.updates_per_step
                prev_optimizer_updates = optimizer_updates
                optimizer_updates += args.updates_per_step
                should_sync_actor_params = (
                    args.async_selfplay
                    and ((prev_optimizer_updates // args.actor_param_sync_updates) != (optimizer_updates // args.actor_param_sync_updates))
                )
                if should_sync_actor_params:
                    if args.cross_process_selfplay:
                        params_host = _extract_host_tree(params, use_pmap=use_pmap)
                        params_host = jax.device_get(params_host)
                        for param_q in actor_params_queues:
                            try:
                                while True:
                                    param_q.get_nowait()
                            except queue_lib.Empty:
                                pass
                            try:
                                param_q.put_nowait(params_host)
                            except queue_lib.Full:
                                pass
                    elif params_ref_lock is not None and params_ref is not None:
                        with params_ref_lock:
                            params_ref[0] = _clone_tree(params)
            else:
                for _ in range(args.updates_per_step):
                    rng_key, sample_key = jax.random.split(rng_key)
                    sample_ids = jax.random.randint(
                        sample_key,
                        shape=(args.batch_size,),
                        minval=0,
                        maxval=replay_count,
                        dtype=jnp.int32,
                    )
                    obs = replay_obs_dev[sample_ids]
                    policy_target = replay_policy_dev[sample_ids]
                    value_target = replay_value_dev[sample_ids]
                    if not args.disable_symmetry_augmentation:
                        rng_key, aug_key = jax.random.split(rng_key)
                        obs, policy_target = _augment_batch_with_random_symmetry_jax(obs, policy_target, aug_key)

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

                    if (
                        args.async_selfplay
                        and (optimizer_updates % args.actor_param_sync_updates == 0)
                    ):
                        if args.cross_process_selfplay:
                            params_host = _extract_host_tree(params, use_pmap=use_pmap)
                            params_host = jax.device_get(params_host)
                            for param_q in actor_params_queues:
                                try:
                                    while True:
                                        param_q.get_nowait()
                                except queue_lib.Empty:
                                    pass
                                try:
                                    param_q.put_nowait(params_host)
                                except queue_lib.Full:
                                    pass
                        elif params_ref_lock is not None and params_ref is not None:
                            with params_ref_lock:
                                params_ref[0] = _clone_tree(params)
            train_updates_ms = (time.perf_counter() - train_updates_start) * 1000.0

            if (
                args.role == "learner"
                and param_sync_snapshot_lock is not None
                and param_sync_snapshot_state is not None
                and (step % args.param_sync_every_steps == 0)
            ):
                snapshot_start = time.perf_counter()
                with param_sync_snapshot_lock:
                    param_sync_snapshot_state["optimizer_updates"] = int(optimizer_updates)
                    param_sync_snapshot_state["blob"] = _build_param_sync_blob(
                        params=params,
                        use_pmap=use_pmap,
                        optimizer_updates=optimizer_updates,
                    )
                param_snapshot_ms = (time.perf_counter() - snapshot_start) * 1000.0

            loss_val = loss_sum / args.updates_per_step
            pol_val = policy_sum / args.updates_per_step
            val_val = value_sum / args.updates_per_step
            lr_val = float(lr_schedule(jnp.asarray(max(optimizer_updates - 1, 0), dtype=jnp.int32)))
            step_ms = (time.perf_counter() - step_start) * 1000.0
            elapsed_sec = max(time.perf_counter() - train_start_time, 1e-6)
            with actor_stats_lock:
                actor_batches_snapshot = actor_batches_total
                actor_examples_snapshot = actor_examples_total
            actor_examples_per_sec = actor_examples_snapshot / elapsed_sec
            actor_batches_per_sec = actor_batches_snapshot / elapsed_sec
            avg_wait_ms = (
                (learner_wait_total_sec / learner_wait_count) * 1000.0 if learner_wait_count > 0 else 0.0
            )
            queue_size = replay_queue.qsize() if replay_queue is not None else 0

            if is_chief:
                details = ""
                if args.detailed_step_log:
                    details = (
                        f" replay_append_ms={replay_append_ms:.1f} train_updates_ms={train_updates_ms:.1f} "
                        f"param_snapshot_ms={param_snapshot_ms:.1f}"
                    )
                print(
                f"step={step} lr={lr_val:.6f} loss={loss_val:.4f} policy={pol_val:.4f} value={val_val:.4f} "
                f"replay={replay_count} new_examples={new_examples} "
                f"bw={black_win} ww={white_win} d={draw} "
                f"step_ms={step_ms:.1f} collect_ms={collect_wait_ms:.1f} wait_avg_ms={avg_wait_ms:.1f} "
                f"qsize={queue_size} actor_batches={actor_batches_snapshot} actor_examples={actor_examples_snapshot} "
                f"actor_bps={actor_batches_per_sec:.2f} actor_eps={actor_examples_per_sec:.1f}{details}"
            )

            if is_chief and args.phase_log_threshold_ms > 0:
                if collect_wait_ms >= args.phase_log_threshold_ms:
                    print(f"phase-slow step={step} phase=collect collect_ms={collect_wait_ms:.1f}")
                if replay_append_ms >= args.phase_log_threshold_ms:
                    print(f"phase-slow step={step} phase=replay_append replay_append_ms={replay_append_ms:.1f}")
                if train_updates_ms >= args.phase_log_threshold_ms:
                    print(
                        f"phase-slow step={step} phase=train_updates train_updates_ms={train_updates_ms:.1f} "
                        f"updates_per_step={args.updates_per_step}"
                    )
                if param_snapshot_ms >= args.phase_log_threshold_ms:
                    print(
                        f"phase-slow step={step} phase=param_snapshot param_snapshot_ms={param_snapshot_ms:.1f} "
                        f"param_sync_every_steps={args.param_sync_every_steps}"
                    )

            if args.arena_every_steps > 0 and (step % args.arena_every_steps == 0):
                arena_start = time.perf_counter()
                rng_key, arena_key = jax.random.split(rng_key)
                if use_pmap:
                    arena_keys = jax.random.split(arena_key, local_devices)
                    assert pmap_arena_step is not None
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
                    best_params = _clone_tree(current_params)
                    if use_pmap:
                        best_params_repl = _replicate_tree_for_pmap(best_params, local_devices=local_devices)
                    best_step = step
                    if is_chief:
                        best_cfg = _checkpoint_config(args, optimizer_updates=optimizer_updates, best_step=best_step)
                        _save_model_checkpoint(best_path, best_params, best_cfg)
                        print(f"arena promoted candidate at step={step} -> {best_path}")
                arena_ms = (time.perf_counter() - arena_start) * 1000.0
                if is_chief and args.phase_log_threshold_ms > 0 and arena_ms >= args.phase_log_threshold_ms:
                    print(f"phase-slow step={step} phase=arena arena_ms={arena_ms:.1f}")

            checkpoint_ms = maybe_save_training_checkpoint(step, "periodic")
            if is_chief and args.phase_log_threshold_ms > 0 and checkpoint_ms >= args.phase_log_threshold_ms:
                print(f"phase-slow step={step} phase=checkpoint checkpoint_ms={checkpoint_ms:.1f}")
    finally:
        if actor_stop is not None:
            actor_stop.set()
        for actor_proc in actor_processes:
            actor_proc.join(timeout=5.0)
            if actor_proc.is_alive():
                actor_proc.terminate()
        for actor_thread in actor_threads:
            actor_thread.join(timeout=5.0)
        if remote_stop_event is not None and remote_client_threads_lock is not None:
            _stop_remote_replay_listener(
                stop_event=remote_stop_event,
                server_sock=remote_server_sock,
                accept_thread=remote_accept_thread,
                client_threads=remote_client_threads,
                client_threads_lock=remote_client_threads_lock,
            )
        if param_sync_stop_event is not None and param_sync_client_threads_lock is not None:
            _stop_param_sync_server(
                stop_event=param_sync_stop_event,
                server_sock=param_sync_server_sock,
                accept_thread=param_sync_accept_thread,
                client_threads=param_sync_client_threads,
                client_threads_lock=param_sync_client_threads_lock,
            )

    final_params = _extract_host_tree(params, use_pmap=use_pmap)
    final_opt_state = _extract_host_tree(opt_state, use_pmap=use_pmap)
    replay_obs_np, replay_policy_np, replay_value_np = _materialize_replay_from_device(
        replay_obs_dev,
        replay_policy_dev,
        replay_value_dev,
        replay_count=replay_count,
    )
    cfg = _checkpoint_config(args, optimizer_updates=optimizer_updates, best_step=best_step)
    if is_chief:
        _save_training_checkpoint(
            args.output,
            params=final_params,
            opt_state=final_opt_state,
            config=cfg,
            step=completed_step,
            optimizer_updates=optimizer_updates,
            rng_key=rng_key,
            np_rng_state=np_rng.bit_generator.state,
            replay_obs=replay_obs_np,
            replay_policy=replay_policy_np,
            replay_value=replay_value_np,
            best_params=best_params,
            best_step=best_step,
        )
        print(f"saved training checkpoint to {args.output}")
        _save_model_checkpoint(best_path, best_params, cfg)
        print(f"saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
