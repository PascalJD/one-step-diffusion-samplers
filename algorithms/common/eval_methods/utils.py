# utils/saving_utils.py
from __future__ import annotations

import json
import numbers
from pathlib import Path
from collections.abc import Sequence

import jax.numpy as jnp
import jax

# Optional W&B import guarded so the utils work without it
try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None


def moving_averages(history: dict[str, Sequence], window_size: int = 5) -> dict[str, list]:
    """Return `{metric_name_mov_avg: [window_mean]}` for each numeric sequence."""
    mov_avgs = {}
    for k, seq in history.items():
        if "mov_avg" in k or not isinstance(seq, Sequence) or len(seq) == 0:
            continue
        try:
            window = jnp.array(seq[-min(len(seq), window_size):])
            mov_avgs[f"{k}_mov_avg"] = [jnp.mean(window, axis=0)]
        except Exception:  # non‑numeric, ignore
            pass
    return mov_avgs


def extract_last_entry(history: dict[str, Sequence]) -> dict[str, float | None]:
    """Return the last value of each sequence (or None if empty)."""
    last = {}
    for k, seq in history.items():
        try:
            last[k] = seq[-1] if len(seq) else None
        except Exception:
            last[k] = None
    return last


def save_samples(cfg, logger: dict, samples: jax.Array) -> None:
    """
    Save `samples` as a .npy under <run-dir>/samples/ only if
    (a) `cfg.save_samples` is True **and**
    (b) the current ELBO is the best so far.
    """
    if not cfg.save_samples:
        return

    elbos = logger.get("KL/elbo", [])
    is_best = len(elbos) <= 1 or elbos[-1] >= max(elbos[:-1])

    if not is_best:
        return

    run_dir = Path(cfg.paths.output)
    out_dir = run_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}.npy"
    out_path = out_dir / fname
    jnp.save(out_path, samples)

    # Optional: upload to W&B as an artifact for long‑term storage
    if cfg.use_wandb and wandb is not None and wandb.run is not None:
        art = wandb.Artifact("best_samples", type="samples")
        art.add_file(str(out_path))
        wandb.run.log_artifact(art)


def compute_reverse_ess(log_weights: jax.Array, eval_samples: int) -> float:
    max_logw = jnp.max(log_weights)
    is_w     = jnp.exp(log_weights - max_logw)
    ess      = (jnp.sum(is_w) ** 2) / (eval_samples * jnp.sum(is_w ** 2))
    return float(ess)


def save_metrics(logger: dict, cfg) -> None:
    """
    Append the latest scalar metrics as one JSON line to
    <run-dir>/metrics/metrics.jsonl and optionally to W&B.
    """
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "metrics.jsonl"

    record = {}
    for k, seq in logger.items():
        if not isinstance(seq, Sequence) or not seq:
            continue
        val = seq[-1]
        if isinstance(val, numbers.Number):
            record[k] = val
        elif hasattr(val, "item"):
            record[k] = float(val.item())
        else:
            # Skip non‑scalar entries (arrays, figures, …)
            continue

    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    if cfg.use_wandb and wandb is not None and wandb.run is not None:
        wandb.log(record, commit=False)  # do not bump the W&B step counter