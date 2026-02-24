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
  --games-per-step 8 \
  --batch-size 256 \
  --num-simulations 32 \
  --output checkpoints/latest.pkl
```

```bash
PYTHONPATH=src python scripts/eval.py --checkpoint checkpoints/latest.pkl --games 20
```

## TPU notes

- Keep `--batch-size` divisible by `jax.local_device_count()` in pmap mode.
- To force single-device fallback: add `--disable-pmap`.
- This is a minimal baseline focused on structure and correctness, not peak throughput.

## Suggested next upgrades

1. Batched self-play actors with `lax.scan` and device sharding.
2. Replay prioritization and better temperature schedule.
3. Arena evaluation and best-model gating.
