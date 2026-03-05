from __future__ import annotations

import argparse
import pickle
import subprocess
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np


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


def _read_pickle(path: str) -> dict[str, Any]:
    if _is_gcs_uri(path):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", path, str(tmp_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            with tmp_path.open("rb") as fp:
                payload = pickle.load(fp)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        with Path(path).open("rb") as fp:
            payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dict")
    return payload


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
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fp:
        pickle.dump(payload, fp)


def _default_output_path(input_path: str) -> str:
    if _is_gcs_uri(input_path):
        bucket, obj = _gcs_object_path(input_path)
        pure = PurePosixPath(obj)
        suffix = pure.suffix
        if suffix:
            new_name = f"{pure.stem}.noreplay{suffix}"
        else:
            new_name = f"{pure.name}.noreplay"
        return f"gs://{bucket}/{pure.with_name(new_name)}"
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}.noreplay{p.suffix}"))


def _as_empty_replay_arrays(payload: dict[str, Any]) -> None:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    board_size = int(config.get("board_size", 15))
    obs_planes = int(config.get("obs_planes", 4))
    num_actions = board_size * board_size

    obs = payload.get("replay_obs")
    policy = payload.get("replay_policy")
    value = payload.get("replay_value")
    horizon = payload.get("replay_horizon")

    if obs is not None:
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 4 and obs_arr.shape[0] >= 0:
            obs_tail = obs_arr.shape[1:]
        else:
            obs_tail = (board_size, board_size, obs_planes)
    else:
        obs_tail = (board_size, board_size, obs_planes)

    if policy is not None:
        pol_arr = np.asarray(policy)
        if pol_arr.ndim == 2 and pol_arr.shape[0] >= 0:
            policy_tail = pol_arr.shape[1:]
        else:
            policy_tail = (num_actions,)
    else:
        policy_tail = (num_actions,)

    payload["replay_obs"] = np.zeros((0, *obs_tail), dtype=np.uint8)
    payload["replay_policy"] = np.zeros((0, *policy_tail), dtype=np.float32)
    payload["replay_value"] = np.zeros((0,), dtype=np.float32)
    payload["replay_horizon"] = np.zeros((0,), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip replay buffers from a gooomoku training checkpoint.")
    parser.add_argument("--input", required=True, help="Input checkpoint path (.pkl), local path or gs://...")
    parser.add_argument(
        "--output",
        default=None,
        help="Output checkpoint path. Default: <input>.noreplay.pkl (or .noreplay suffix).",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite input checkpoint.")
    parser.add_argument(
        "--mode",
        choices=("drop", "empty"),
        default="drop",
        help="drop: remove replay_* keys; empty: keep replay_* keys but set them to empty arrays.",
    )
    args = parser.parse_args()

    input_path = str(args.input)
    if args.in_place and args.output:
        raise ValueError("cannot set both --in-place and --output")
    output_path = input_path if args.in_place else (str(args.output) if args.output else _default_output_path(input_path))

    payload = _read_pickle(input_path)
    old_examples = int(np.asarray(payload.get("replay_value", np.zeros((0,), dtype=np.float32))).shape[0])

    if args.mode == "drop":
        payload.pop("replay_obs", None)
        payload.pop("replay_policy", None)
        payload.pop("replay_value", None)
        payload.pop("replay_horizon", None)
    else:
        _as_empty_replay_arrays(payload)

    _write_pickle(output_path, payload)
    print(
        f"strip_replay done mode={args.mode} input={input_path} output={output_path} "
        f"old_replay_examples={old_examples}"
    )


if __name__ == "__main__":
    main()
