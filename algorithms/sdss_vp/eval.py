# algorithms/sdss_vp/eval.py
from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Any
import time
import csv
import json

import numpy as np
import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import wandb

from algorithms.common.eval_methods.utils import (
    moving_averages, save_samples, compute_reverse_ess
)
from algorithms.common.ipm_eval import discrepancies

def plot_paths(full_paths, *, max_paths=64, wandb_key="figures/paths"):
    arr = np.asarray(jax.device_get(full_paths))
    if arr.ndim == 2:
        arr = arr[None, ...]
    bsz, t1, d = arr.shape
    t = np.arange(t1)

    if bsz > max_paths:
        idx = np.linspace(0, bsz - 1, max_paths, dtype=int)
        arr = arr[idx]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for p in arr:
        ax.plot(t, p[:, 0], lw=0.8, alpha=0.8)
    ax.set_xlabel("integration step")
    ax.set_ylabel(r"$x_0$")
    ax.set_title("Trajectories (first coordinate)")

    wb_img = wandb.Image(fig)
    plt.close(fig)
    return {wandb_key: [wb_img]}


def get_multi_eval_fn(
    rnd_base_ode,
    rnd_base_sde,
    target,
    target_samples,
    cfg,
    eval_budgets,
    viz_budget,
    n_repeats: int | None = None, 
):
    # Determine number of simulation repeats for aggregation
    if n_repeats is None:
        n_repeats = int(getattr(cfg, "n_simulations", 20))
    n_repeats = max(1, int(n_repeats))

    # Whether we can sample from target (for forward/EUBO metrics)
    target_can_sample: bool = bool(getattr(target, "can_sample", False))

    # JIT one reverse/forward per budget; ensure we get paths back
    rnd_rev_ode = {
        k: jax.jit(partial(
            rnd_base_ode, prior_to_target=True, eval_steps=k, return_traj=True
        ))
        for k in eval_budgets
    }
    rnd_rev_sde = {
        k: jax.jit(partial(
            rnd_base_sde, prior_to_target=True, eval_steps=k, return_traj=True
        ))
        for k in eval_budgets
    }
    # Forward passes only if target can sample
    if target_can_sample:
        rnd_fwd_ode = {
            k: jax.jit(partial(
                rnd_base_ode, prior_to_target=False, eval_steps=k, return_traj=True
            ))
            for k in eval_budgets
        }
        rnd_fwd_sde = {
            k: jax.jit(partial(
                rnd_base_sde, prior_to_target=False, eval_steps=k, return_traj=True
            ))
            for k in eval_budgets
        }
    else:
        rnd_fwd_ode = {}
        rnd_fwd_sde = {}

    # Logger collects scalars (we'll store aggregated means, and also *_std)
    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "logZ/detflow": [],
        "logZ/delta_detflow": [],
        "ESS/detflow": [],
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "other/target_log_prob": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def _suffix(key, k):
        return f"{key}@k={k}"

    def _append(dct, key, val):
        dct.setdefault(key, []).append(val)

    def _accumulate(acc: dict[str, list], key: str, value):
        acc.setdefault(key, []).append(_to_python(value))

    def eval_once(model_state, rng):
        samples_for_plot = None

        for k in eval_budgets:
            # Accumulators for this k across n_repeats
            acc: dict[str, list] = {}

            for rep in range(n_repeats):
                # ODE reverse pass: samples for discrepancies/visuals
                rng, sub = jax.random.split(rng)
                (samples_ode, _rc_ode, logdet_ode, term_c_ode, paths_ode) = rnd_rev_ode[k](
                    sub, model_state, model_state.params
                )

                # Deterministic-flow weights and logZ (PF-ODE)
                log_w_det = -term_c_ode + logdet_ode
                ln_z_det = jax.scipy.special.logsumexp(log_w_det) - jnp.log(cfg.eval_samples)
                ess_det = compute_reverse_ess(log_w_det, cfg.eval_samples)
                elbo_det = jnp.mean(log_w_det)

                _accumulate(acc, "logZ/detflow", ln_z_det)
                _accumulate(acc, "ESS/detflow", ess_det)
                _accumulate(acc, "KL/elbo_detflow", elbo_det)

                # Deterministic forward (EUBO) & forward logZ via PF-ODE forward
                if target_can_sample:
                    rng, sub = jax.random.split(rng)
                    (_, _rc_fwd_ode, logdet_fwd_ode, term_c_fwd_ode, _paths_fwd_ode) = rnd_fwd_ode[k](
                        sub, model_state, model_state.params
                    )
                    log_w_det_fwd = -term_c_fwd_ode + logdet_fwd_ode
                    eubo_det = jnp.mean(log_w_det_fwd)
                    ln_z_det_fwd = -(jax.scipy.special.logsumexp(-log_w_det_fwd) - jnp.log(cfg.eval_samples))

                    _accumulate(acc, "KL/eubo_detflow", eubo_det)
                    _accumulate(acc, "logZ/forward_detflow", ln_z_det_fwd)

                # Discrepancies on ODE samples (only if we have target_samples)
                if target_samples is not None:
                    for d in cfg.discrepancies:
                        key_d = f"discrepancies/{d}"
                        val_d = getattr(discrepancies, f"compute_{d}")(
                            target_samples, samples_ode, cfg
                        )
                        _accumulate(acc, key_d, val_d)

                # Keep one set of visuals for viz_budget (last repeat)
                if k == viz_budget and rep == (n_repeats - 1):
                    logger.update(plot_paths(paths_ode, wandb_key="figures/paths"))
                    _append(logger, "other/target_log_prob",
                            jnp.mean(target.log_prob(samples_ode)))
                    samples_for_plot = samples_ode

                # SDE reverse pass (EM integrator)
                rng, sub = jax.random.split(rng)
                (_, run_c_em, stoch_c, term_c, _paths_sde) = rnd_rev_sde[k](
                    sub, model_state, model_state.params
                )

                log_w_em = -(run_c_em + stoch_c + term_c)
                ln_z_em = jax.scipy.special.logsumexp(log_w_em) - jnp.log(cfg.eval_samples)
                elbo_em = jnp.mean(log_w_em)
                ess_em = compute_reverse_ess(log_w_em, cfg.eval_samples)

                _accumulate(acc, "logZ/reverse", ln_z_em)
                _accumulate(acc, "KL/elbo", elbo_em)
                _accumulate(acc, "ESS/reverse", ess_em)

                # SDE forward pass (EM integrator)
                if target_can_sample:
                    rng, sub = jax.random.split(rng)
                    (_, run_c_fwd, stoch_c_fwd, term_c_fwd, _) = rnd_fwd_sde[k](sub, model_state, model_state.params)
                    fwd_log_w = -(run_c_fwd + stoch_c_fwd + term_c_fwd)
                    eubo = jnp.mean(fwd_log_w)
                    ln_z_fwd = -(jax.scipy.special.logsumexp(-fwd_log_w) - jnp.log(cfg.eval_samples))
                    ess_fwd = jnp.exp(ln_z_fwd - (jax.scipy.special.logsumexp(fwd_log_w) - jnp.log(cfg.eval_samples)))

                    _accumulate(acc, "KL/eubo", eubo)
                    _accumulate(acc, "logZ/forward", ln_z_fwd)
                    _accumulate(acc, "ESS/forward", ess_fwd)

                    # GT logZ deltas: only meaningful when sampling forward exists
                    if k == viz_budget and (target.log_Z is not None):
                        _accumulate(acc, "logZ/delta_forward", jnp.abs(ln_z_fwd - target.log_Z))
                        _accumulate(acc, "logZ/delta_forward_detflow", jnp.abs(ln_z_det_fwd - target.log_Z))

                # Reverse deltas (if GT available)
                if k == viz_budget and (target.log_Z is not None):
                    _accumulate(acc, "logZ/delta_detflow", jnp.abs(ln_z_det - target.log_Z))
                    _accumulate(acc, "logZ/delta_reverse", jnp.abs(ln_z_em - target.log_Z))

            # Aggregate (mean/std) for this k and push to logger
            def _mean_std(vals: list) -> tuple[float, float]:
                arr = np.asarray(vals, dtype=float)
                return float(arr.mean()), float(arr.std())

            # Iterate over all accumulated metrics for this k
            for base_key, values in sorted(acc.items()):
                mean_val, std_val = _mean_std(values)
                _append(logger, _suffix(base_key, k), mean_val)
                _append(logger, _suffix(f"{base_key}_std", k), std_val)

                # For viz_budget, also expose unsuffixed keys (means) plus *_std for convenience
                if k == viz_budget:
                    _append(logger, base_key, mean_val)
                    _append(logger, f"{base_key}_std", std_val)

        # Visuals and extras for viz_budget
        if samples_for_plot is not None:
            logger.update(
                target.visualise(samples=samples_for_plot, show=cfg.visualize_samples)
            )
        if cfg.compute_emc and cfg.target.has_entropy:
            _append(logger, "other/EMC", target.entropy(samples_for_plot))

        if cfg.moving_average.use_ma:
            logger.update(
                moving_averages(
                    logger, window_size=cfg.moving_average.window_size
                )
            )
        if cfg.save_samples and (samples_for_plot is not None):
            save_samples(cfg, logger, samples_for_plot)

        return logger

    return eval_once, logger