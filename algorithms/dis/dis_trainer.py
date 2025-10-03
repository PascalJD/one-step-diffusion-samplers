"""
Time-Reversed Diffusion Sampler (DIS)
For further details see https://openreview.net/pdf?id=oYIjw37pTP
"""

from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import wandb
import logging

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.dis.dis_rnd import neg_elbo as neg_elbo_line
from algorithms.dis.dis_rnd import rnd as rnd_line
from algorithms.dis.dis_is_weights import neg_elbo as neg_elbo_is
from algorithms.dis.dis_is_weights import rnd as rnd_is
from utils.print_util import print_results

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


def dis_trainer(cfg, target, checkpointer):
    manager = None
    if isinstance(checkpointer, dict):
        manager = checkpointer.get("last")
        if manager is None and checkpointer:
            manager = next(iter(checkpointer.values()))
    elif checkpointer is not None:
        manager = checkpointer

    if manager is None:
        log.info("Checkpointing disabled, no checkpoint manager provided.")

    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Define initial and target density
    initial_density = distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                                     jnp.ones(dim) * alg_cfg.init_std)
    aux_tuple = (alg_cfg.init_std, initial_density.sample, initial_density.log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    noise_schedule = alg_cfg.noise_schedule

    objective = neg_elbo_line
    rnd = rnd_line
    if alg_cfg.loss == "is":
        objective = neg_elbo_is
        rnd = rnd_is
    print(f"loss is {alg_cfg.loss}")

    loss = jax.jit(jax.grad(objective, 2, has_aux=True), static_argnums=(3, 4, 5, 6, 7))
    rnd_short = partial(rnd, batch_size=cfg.eval_samples, aux_tuple=aux_tuple,
                        target=target, num_steps=cfg.algorithm.num_steps,
                        noise_schedule=noise_schedule, stop_grad=True, use_ode=True)

    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()
        grads, _ = loss(key, model_state, model_state.params, alg_cfg.batch_size,
                        aux_tuple, target, alg_cfg.num_steps, noise_schedule)
        timer += time() - iter_time

        model_state = model_state.apply_gradients(grads=grads)

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
            
            # Checkpoint  
            if step > 0 and manager is not None:
                pkg = dict(
                    model_state=jax.device_get(model_state),
                    key_gen=key_gen,
                    step=step,
                    timer=timer,
                )
                future = manager.save(step, pkg)
                print(f"[Orbax] step={step} saved (queued={future})")