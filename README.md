# gooomoku (minimal JAX + mctx skeleton)

This is a minimal self-play Gomoku scaffold using:
- JAX
- Flax
- Optax
- mctx (Gumbel MuZero search)

It includes:
- Environment: `src/gooomoku/env.py`
- Policy/Value network: `src/gooomoku/net.py`
- mctx bridge: `src/gooomoku/mctx_adapter.py`
- Self-play: `scripts/self_play.py`
- Training: `scripts/train.py`
- Evaluation: `scripts/eval.py`

## Colab setup (TPU v5e-8)

Run this in Colab (TPU runtime):

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
```

Then verify TPU cores:

```bash
python -c "import jax; print('devices:', jax.local_device_count(), jax.devices())"
```

`scripts/train.py` will automatically enable `pmap` when `jax.local_device_count() > 1`.
For a v5e-8 runtime, this should map to 8 local cores.

## Quick start

```bash
PYTHONPATH=src python scripts/self_play.py --board-size 9 --num-simulations 32
```

```bash
PYTHONPATH=src python scripts/train.py \
  --board-size 9 \
  --train-steps 20 \
  --games-per-step 16 \
  --updates-per-step 8 \
  --batch-size 256 \
  --num-simulations 32 \
  --max-num-considered-actions 24 \
  --arena-every-steps 10 \
  --arena-games 64 \
  --arena-replace-threshold 0.55 \
  --compute-dtype bfloat16 \
  --param-dtype float32 \
  --lr 1e-3 \
  --lr-warmup-steps 100 \
  --lr-end-value 1e-4 \
  --output checkpoints/latest.pkl
```

```bash
PYTHONPATH=src python scripts/eval.py --checkpoint checkpoints/latest.pkl --games 20
```

Resume from checkpoint:

```bash
PYTHONPATH=src python scripts/train.py \
  --resume-from checkpoints/latest.pkl \
  --train-steps 200 \
  --output checkpoints/latest.pkl
```

Native Cloud Storage checkpoints are also supported (without gcsfuse mount) by using `gs://` paths:

```bash
PYTHONPATH=src python scripts/train.py \
  --resume-from gs://your-bucket/checkpoints/latest.pkl \
  --output gs://your-bucket/checkpoints/latest.pkl
```

When using `gs://` paths, `gcloud` CLI must be available on the runtime host.

## TPU notes

- Keep `--batch-size` divisible by `jax.local_device_count()` in pmap mode.
- To force single-device fallback: add `--disable-pmap`.
- For multi-host TPU slices (for example v4-16 with 2 workers), launch `scripts/train.py` on **all workers** and keep `--distributed-init auto` (default) or set `--distributed-init on` to require `jax.distributed.initialize()`.
- For asynchronous actor-learner on TPU slices, add `--async-selfplay --cross-process-selfplay` to run self-play actors in separate processes and avoid TPU runtime conflicts from threaded actor execution.
- Training checkpoint writes are chief-only (`process_index == 0`) in distributed mode to avoid multi-host file write races.
- This is a minimal baseline focused on structure and correctness, not peak throughput.

### Role-separated learner/actor mode (TPU actors)

`scripts/train.py` now supports `--role all|learner|actor` (default `all`).

- `--role learner`: receives self-play batches from network (`--replay-host/--replay-port`) and trains.
- `--role actor`: runs self-play and pushes batches to learner over TCP; can use TPU in-process (`pmap`) because it is no longer cross-process in the same VM.

Example split deployment:

```bash
# learner host
PYTHONPATH=src python scripts/train.py \
  --role learner \
  --replay-host 0.0.0.0 \
  --replay-port 19091 \
  --train-steps 500 \
  --batch-size 1024 \
  --updates-per-step 16 \
  --checkpoint-every-steps 20 \
  --output checkpoints/latest.pkl

# actor host (TPU-enabled)
PYTHONPATH=src python scripts/train.py \
  --role actor \
  --jax-platforms tpu \
  --replay-host <learner-ip> \
  --replay-port 19091 \
  --selfplay-batch-games 128 \
  --num-simulations 64 \
  --resume-from checkpoints/latest.pkl \
  --actor-sync-every-batches 8
```

Notes:
- Actor role syncs parameters by reloading `--resume-from` checkpoint periodically.
- `--replay-host` can be wildcard (`0.0.0.0`) for learner bind, but actor must use a routable learner IP/hostname.
- Existing `--role all` behavior is unchanged.

## Suggested next upgrades

1. Batched self-play actors with `lax.scan` and device sharding.
2. Replay prioritization and better temperature schedule.
3. Arena evaluation and best-model gating.

## Web PvAI platform (JAX GPU inference)

Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU inference, install CUDA-enabled JAX wheel first (example for CUDA 12):

```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run web service (model artifact can be `.pkl` or `.tar.gz` containing a `.pkl` checkpoint):

```bash
PYTHONPATH=src python scripts/web_play.py \
  --model-artifact "/home/shenmo/Downloads/sb (2).tar.gz" \
  --board-size 15 \
  --channels 96 \
  --blocks 8 \
  --num-simulations 256 \
  --max-num-considered-actions 64 \
  --compute-dtype bfloat16 \
  --param-dtype float32 \
  --host 127.0.0.1 \
  --port 8000
```

Optional: verify artifact integrity with SHA256:

```bash
sha256sum "/home/shenmo/Downloads/sb (2).tar.gz"
PYTHONPATH=src python scripts/web_play.py \
  --model-artifact "/home/shenmo/Downloads/sb (2).tar.gz" \
  --artifact-sha256 "<sha256-of-artifact>"
```

Then open:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/api/health
```

### Parameter mapping from your training command

Your training command contains these architecture/search arguments that must match for inference:

- `--board-size 15`
- `--channels 96`
- `--blocks 8`
- `--num-simulations 256`
- `--max-num-considered-actions 64`
- `--compute-dtype bfloat16`
- `--param-dtype float32`

`scripts/web_play.py` will prefer checkpoint `config` values if present, otherwise it falls back to the CLI values above.

Security note: model artifacts are loaded via pickle payload schema; only use trusted artifacts. You can enable SHA256 verification with `--artifact-sha256`.
