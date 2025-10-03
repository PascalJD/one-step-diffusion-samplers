from functools import partial
from pathlib import Path
import csv
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb

from algorithms.common.eval_methods.utils import moving_averages, save_samples, compute_reverse_ess
from algorithms.common.ipm_eval import discrepancies


def get_eval_fn(
        rnd,
        target,
        target_samples,
        cfg):

    rnd_reverse = jax.jit(partial(rnd, prior_to_target=True))

    with_fwd = bool(getattr(cfg, "compute_forward_metrics", False) and getattr(target, "can_sample", False))
    if with_fwd:
        if cfg.algorithm.name == "sdss_vp":
            rnd_forward = jax.jit(partial(rnd, prior_to_target=False, use_ode=cfg.algorithm.eval_ode))
        else:
            rnd_forward = jax.jit(partial(rnd, prior_to_target=False))

    logger: Dict[str, List] = {
        'KL/elbo': [],
        'KL/eubo': [],
        'logZ/delta_forward': [],
        'logZ/forward': [],
        'logZ/delta_reverse': [],
        'logZ/reverse': [],
        'ESS/forward': [],
        'ESS/reverse': [],
        'discrepancies/mmd': [],
        'discrepancies/sd': [],
        'other/target_log_prob': [],
        'other/EMC': [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def _accumulate(acc: Dict[str, List[float]], k: str, v) -> None:
        acc.setdefault(k, []).append(float(np.asarray(v)))

    def short_eval(model_state, key):
        # Support single-model or tuple (two models) state
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)

        n_repeats = int(getattr(cfg, "n_repeats", getattr(cfg, "n_simulations", 20)))
        n_repeats = max(1, n_repeats)

        acc: Dict[str, List[float]] = {}
        samples_for_viz = None 

        for _ in range(n_repeats):
            key, sub_rev, sub_fwd = jax.random.split(key, 3)

            samples, running_costs, stochastic_costs, terminal_costs = rnd_reverse(sub_rev, model_state, *params)
            log_is_weights = -(running_costs + stochastic_costs + terminal_costs)
            ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
            elbo = -jnp.mean(running_costs + terminal_costs)
            ess_rev = compute_reverse_ess(log_is_weights, cfg.eval_samples)

            _accumulate(acc, "logZ/reverse", ln_z)
            _accumulate(acc, "KL/elbo", elbo)
            _accumulate(acc, "ESS/reverse", ess_rev)
            _accumulate(acc, "other/target_log_prob", jnp.mean(target.log_prob(samples)))

            if target.log_Z is not None:
                _accumulate(acc, "logZ/delta_reverse", jnp.abs(ln_z - target.log_Z))

            if with_fwd:
                fwd_samples, fwd_running_costs, fwd_stochastic_costs, fwd_terminal_costs = rnd_forward(
                    sub_fwd, model_state, *params
                )
                fwd_log_is_weights = -(fwd_running_costs + fwd_stochastic_costs + fwd_terminal_costs)
                fwd_ln_z = -(jax.scipy.special.logsumexp(-fwd_log_is_weights) - jnp.log(cfg.eval_samples))
                fwd_ess = jnp.exp(fwd_ln_z - (jax.scipy.special.logsumexp(fwd_log_is_weights) - jnp.log(cfg.eval_samples)))
                eubo = jnp.mean(fwd_log_is_weights)

                _accumulate(acc, "logZ/forward", fwd_ln_z)
                _accumulate(acc, "KL/eubo", eubo)
                _accumulate(acc, "ESS/forward", fwd_ess)

                if target.log_Z is not None:
                    _accumulate(acc, "logZ/delta_forward", jnp.abs(fwd_ln_z - target.log_Z))

            if target_samples is not None:
                for d in cfg.discrepancies:
                    val = getattr(discrepancies, f'compute_{d}')(target_samples, samples, cfg)
                    _accumulate(acc, f'discrepancies/{d}', val)

            samples_for_viz = samples

            if getattr(cfg, "compute_emc", False) and getattr(cfg.target, "has_entropy", False):
                _accumulate(acc, "other/EMC", target.entropy(samples))

        for k_acc, values in sorted(acc.items()):
            arr = np.asarray(values, dtype=float)
            mu, sd = float(arr.mean()), float(arr.std())
            logger.setdefault(k_acc, []).append(mu)
            logger.setdefault(f"{k_acc}_std", []).append(sd)

        if samples_for_viz is not None:
            logger.update(target.visualise(samples=samples_for_viz, show=cfg.visualize_samples))
            if getattr(cfg, "save_samples", False):
                save_samples(cfg, logger, samples_for_viz)

        if cfg.moving_average.use_ma:
            logger.update(moving_averages(logger, window_size=cfg.moving_average.window_size))

        return logger

    return short_eval, logger