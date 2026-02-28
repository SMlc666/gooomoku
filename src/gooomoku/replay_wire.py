from __future__ import annotations

import hashlib
import json
import socket
import struct
import time
from typing import Final

import numpy as np

MAGIC: Final[bytes] = b"GKB1"
HEADER_LEN_STRUCT: Final[struct.Struct] = struct.Struct("!I")
FRAME_PREFIX_STRUCT: Final[struct.Struct] = struct.Struct("!4sI")
_OBS_DTYPE = np.dtype(np.uint8)
_POLICY_DTYPE = np.dtype(np.float32)
_VALUE_DTYPE = np.dtype(np.float32)


class ReplayWireError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ReplayWireError("socket closed while receiving replay frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ensure_contiguous(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(a, dtype=dtype)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def encode_selfplay_batch(
    obs: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    black_win: int,
    white_win: int,
    draw: int,
    valid_count: int,
) -> bytes:
    obs_arr = _ensure_contiguous(obs, _OBS_DTYPE)
    policy_arr = _ensure_contiguous(policy, _POLICY_DTYPE)
    value_arr = _ensure_contiguous(value, _VALUE_DTYPE)

    if obs_arr.shape[0] != policy_arr.shape[0] or obs_arr.shape[0] != value_arr.shape[0]:
        raise ReplayWireError("obs/policy/value first dimension mismatch")

    obs_bytes = obs_arr.tobytes(order="C")
    policy_bytes = policy_arr.tobytes(order="C")
    value_bytes = value_arr.tobytes(order="C")
    body = obs_bytes + policy_bytes + value_bytes

    obs_offset = 0
    policy_offset = len(obs_bytes)
    value_offset = len(obs_bytes) + len(policy_bytes)
    checksum = hashlib.sha256(body).hexdigest()

    header = {
        "schema": "gooomoku-selfplay-v1",
        "obs": {
            "dtype": str(obs_arr.dtype),
            "shape": list(obs_arr.shape),
            "offset": obs_offset,
            "nbytes": len(obs_bytes),
        },
        "policy": {
            "dtype": str(policy_arr.dtype),
            "shape": list(policy_arr.shape),
            "offset": policy_offset,
            "nbytes": len(policy_bytes),
        },
        "value": {
            "dtype": str(value_arr.dtype),
            "shape": list(value_arr.shape),
            "offset": value_offset,
            "nbytes": len(value_bytes),
        },
        "black_win": int(black_win),
        "white_win": int(white_win),
        "draw": int(draw),
        "valid_count": int(valid_count),
        "checksum_sha256": checksum,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = FRAME_PREFIX_STRUCT.pack(MAGIC, len(header_bytes))
    return prefix + header_bytes + body


def decode_selfplay_batch(frame: bytes):
    if len(frame) < FRAME_PREFIX_STRUCT.size:
        raise ReplayWireError("frame too short")
    magic, header_len = FRAME_PREFIX_STRUCT.unpack(frame[: FRAME_PREFIX_STRUCT.size])
    if magic != MAGIC:
        raise ReplayWireError("invalid frame magic")

    header_start = FRAME_PREFIX_STRUCT.size
    header_end = header_start + int(header_len)
    if header_end > len(frame):
        raise ReplayWireError("invalid header length")

    header = json.loads(frame[header_start:header_end].decode("utf-8"))
    if header.get("schema") != "gooomoku-selfplay-v1":
        raise ReplayWireError(f"unsupported schema: {header.get('schema')}")

    body = frame[header_end:]
    checksum = hashlib.sha256(body).hexdigest()
    if checksum != header.get("checksum_sha256"):
        raise ReplayWireError("frame checksum mismatch")

    def decode_array(spec: dict, expected_dtype: np.dtype) -> np.ndarray:
        offset = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        shape = tuple(int(x) for x in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        if dtype != expected_dtype:
            raise ReplayWireError(f"unexpected dtype: {dtype} != {expected_dtype}")
        if offset < 0 or nbytes < 0 or (offset + nbytes) > len(body):
            raise ReplayWireError("array bounds out of range")
        raw = memoryview(body)[offset : offset + nbytes]
        arr = np.frombuffer(raw, dtype=dtype)
        expected_elems = int(np.prod(shape, dtype=np.int64))
        if arr.size != expected_elems:
            raise ReplayWireError("array element count mismatch")
        return np.asarray(arr.reshape(shape), copy=True)

    obs = decode_array(header["obs"], _OBS_DTYPE)
    policy = decode_array(header["policy"], _POLICY_DTYPE)
    value = decode_array(header["value"], _VALUE_DTYPE)

    if obs.shape[0] != policy.shape[0] or obs.shape[0] != value.shape[0]:
        raise ReplayWireError("decoded obs/policy/value first dimension mismatch")

    return (
        obs,
        policy,
        value,
        int(header["black_win"]),
        int(header["white_win"]),
        int(header["draw"]),
        int(header["valid_count"]),
    )


def send_selfplay_batch(sock: socket.socket, batch) -> None:
    frame = encode_selfplay_batch(*batch)
    sock.sendall(HEADER_LEN_STRUCT.pack(len(frame)))
    sock.sendall(frame)


def recv_selfplay_batch(sock: socket.socket):
    (frame_len,) = HEADER_LEN_STRUCT.unpack(_recv_exact(sock, HEADER_LEN_STRUCT.size))
    if frame_len <= 0:
        raise ReplayWireError("invalid frame length")
    frame = _recv_exact(sock, frame_len)
    return decode_selfplay_batch(frame)


def connect_with_retry(host: str, port: int, retry_seconds: float) -> socket.socket:
    delay = max(0.1, float(retry_seconds))
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            return sock
        except OSError:
            sock.close()
            time.sleep(delay)
