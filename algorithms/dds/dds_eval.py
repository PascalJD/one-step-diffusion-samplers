# algorithms/dis/dis_eval.py
import os, json, logging
from pathlib import Path
from collections import defaultdict
from functools import partial
from numbers import Number

import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import distrax

from algorithms.dds.dds_rnd import rnd
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn

log = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _is_scalar(x):
    if isinstance(x, Number):
        return True
    if isinstance(x, (np.ndarray, jax.Array)) and x.ndim == 0:
        return True
    return False

def dds_eval(cfg: DictConfig, model_state, step: int) -> None:
    eval_cfg = cfg.get("eval", {})
    n_repeat = int(eval_cfg.get("n_repeat", 1))
    alg_cfg = cfg.algorithm

    target = cfg.target.fn
    dim = target.dim
    initial_density = distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                                     jnp.ones(dim) * alg_cfg.init_std)
    aux_tuple = (alg_cfg.init_std, initial_density.sample, initial_density.log_prob, alg_cfg.noise_scale)
    target_samples = target.sample(jax.random.PRNGKey(0), (eval_cfg.batch_size,))
    rnd_short = partial(rnd, batch_size=cfg.eval_samples, initial_density_tuple=aux_tuple,
                        target=target, num_steps=cfg.algorithm.num_steps,
                        noise_schedule=cfg.algorithm.noise_schedule, stop_grad=True)
    eval_fn, _ = get_eval_fn(rnd_short, target, target_samples, cfg)

    metrics = defaultdict(list)
    key_gen = jax.random.PRNGKey(cfg.seed)

    print(f"[DIS EVAL] step={step}  repeats={n_repeat}")
    for rep in range(n_repeat):
        key, key_gen = jax.random.split(key_gen)
        logdict = eval_fn(model_state, key)
        for name, series in logdict.items():
            if series:
                v = series[-1]
                if _is_scalar(v):
                    metrics[name].append(float(v))

    # Save
    rows = []
    for m, vals in metrics.items():
        arr = np.asarray(vals, float)
        rows.append({
            "metric": m,
            "mean": arr.mean(),
            "std": arr.std(ddof=1) if len(arr) > 1 else np.nan,
            "n": len(arr),
            "step": step,
        })
    df = pd.DataFrame(rows).sort_values(["metric", "step"])

    out_dir = Path("results") / cfg.algorithm.name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{cfg.target.name}_seed{cfg.seed}_step{step}"
    df.to_csv(out_dir / f"{base}.csv", index=False)
    with open(out_dir / f"{base}.raw.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in metrics.items()}, f, indent=2)
    print(f"[DIS EVAL] wrote {out_dir / f'{base}.csv'}")