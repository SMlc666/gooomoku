from __future__ import annotations

import os
from pathlib import Path

import jax


def configure_jax_runtime(*, app_name: str, repo_root: Path | None = None) -> Path | None:

    if os.environ.get("GOOOMOKU_LOG_COMPILES", "0") == "1":
        jax.config.update("jax_log_compiles", True)

    if os.environ.get("GOOOMOKU_ENABLE_JAX_CACHE", "1") != "1":
        return None

    if "GOOOMOKU_JAX_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["GOOOMOKU_JAX_CACHE_DIR"]).expanduser()
    else:
        if repo_root is not None:
            cache_dir = repo_root / ".jax_cache"
        else:
            cache_dir = Path.home() / ".cache" / "gooomoku" / "jax_compile"

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(str(cache_dir))
    except Exception as exc:
        print(f"[{app_name}] warning: failed to enable JAX compilation cache at {cache_dir}: {exc}")
        return None

    return cache_dir
