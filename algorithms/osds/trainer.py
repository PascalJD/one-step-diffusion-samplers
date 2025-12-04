from functools import partial
from time import time
import logging

import distrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.scipy as jsp
import wandb
import orbax.checkpoint as ocp

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.osds.fb_rnd import rnd
from algorithms.osds.distill import distillation_loss
from algorithms.osds.eval import get_multi_eval_fn
from utils.print_util import print_results

log = logging.getLogger(__name__)
logging.getLogger("orbax.checkpoint").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


def trainer(cfg, target, checkpointer):
    if checkpointer is None:
        checkpoint_managers = {}
    elif isinstance(checkpointer, ocp.CheckpointManager):
        checkpoint_managers = {"last": checkpointer}
    elif isinstance(checkpointer, dict):
        checkpoint_managers = checkpointer
    else:
        raise TypeError("Expected an Orbax CheckpointManager, dict of managers, or None.")

    if not checkpoint_managers:
        log.info("Checkpointing disabled – no checkpoint manager provided.")

    rng = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size

    prior = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    prior_tuple = (alg_cfg.init_std, prior.sample, prior.log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    rng, sub = jax.random.split(rng)
    model_state = init_model(sub, dim, alg_cfg)

    ema_params = model_state.params

    schedule = alg_cfg.noise_schedule
    num_steps = alg_cfg.num_steps

    sd_warmup = alg_cfg.sd_warmup
    sd_wmax = alg_cfg.sd_wmax
    ema_decay = alg_cfg.ema_decay

    integ_train = getattr(alg_cfg, "integrator_train", "em")
    integ_eval = getattr(alg_cfg, "integrator_eval", "em")

    eval_budgets = getattr(
        alg_cfg, "multi_eval_steps", [num_steps]
    )
    eval_repetitions = getattr(
        cfg, "eval_repetitions", 20
    )
    viz_budget = getattr(alg_cfg, "viz_eval_steps", max(eval_budgets))
    for k in eval_budgets:
        if num_steps % k != 0:
            raise ValueError(f"num_steps={num_steps} not divisible by k={k}")

    def sampling_loss(rng, state, params):
        rng, sub = jax.random.split(rng)
        final_x, run_c, stoch_c, term_c, traj = rnd(
            sub,
            state,
            params,
            batch_size,
            prior_tuple,
            target,
            num_steps,
            schedule,
            stop_grad=False,
            return_traj=True,
            use_ode=False,
            integrator_kind=integ_train,
            schedule_type=schedule_type, 
            beta_min=bmin, 
            beta_max=bmax
        )
        loss = jnp.mean(run_c + term_c)
        return loss, (rng, traj)

    grad_sampling = jax.jit(
        jax.value_and_grad(sampling_loss, argnums=2, has_aux=True)
    )

    def sd_loss(rng, state, params, teacher_params, paths):
        rng, sub = jax.random.split(rng)
        mse = distillation_loss(
            sub,
            state,
            params,
            teacher_params,
            paths,
            batch_size,
            prior_tuple,
            target,
            schedule,
            num_steps,
            trace_weight=alg_cfg.trace_weight,
            n_trace_probes=alg_cfg.n_trace_probes_train,
            jac_weight=alg_cfg.jac_weight,
        )
        return jnp.mean(mse), rng

    grad_sd = jax.jit(jax.value_and_grad(sd_loss, argnums=2, has_aux=True))

    schedule_type = getattr(alg_cfg, "schedule_type", "linear")
    bmin = bmax = None
    if schedule_type.lower() == "linear":
        bmin = alg_cfg.beta_min
        bmax = alg_cfg.beta_max

    # Eval with two RND bases: ODE for samples; SDE for weights
    rnd_base_ode = partial(
        rnd,
        batch_size=cfg.eval_samples,
        prior_tuple=prior_tuple,
        target=target,
        num_steps=alg_cfg.num_steps,
        noise_schedule=alg_cfg.noise_schedule,
        stop_grad=True,
        use_ode=True,
        return_traj=True,
        schedule_type=schedule_type,
        beta_min=bmin,
        beta_max=bmax,
        n_trapz=1025,
        integrator_kind=integ_train, 
    )
    rnd_base_sde = partial(
        rnd,
        batch_size=cfg.eval_samples,
        prior_tuple=prior_tuple,
        target=target,
        num_steps=alg_cfg.num_steps,
        noise_schedule=alg_cfg.noise_schedule,
        stop_grad=True,
        use_ode=False,
        return_traj=True,
        schedule_type=schedule_type,
        beta_min=bmin,
        beta_max=bmax,
        n_trapz=1025,
        integrator_kind=integ_eval,      
    )
    eval_fn, logger = get_multi_eval_fn(
        rnd_base_ode=rnd_base_ode,
        rnd_base_sde=rnd_base_sde,
        target=target,
        target_samples=target_samples,
        cfg=cfg,
        eval_budgets=eval_budgets,
        viz_budget=viz_budget
    )   

    logger["train/sc"] = []
    logger["train/grad_diff"] = []
    logger["train/grad_sd"] = []
    logger["train/sd"] = []
    logger["KL/neg_elbo"] = []
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    def sd_weight(step):
        return sd_wmax * jnp.minimum(1.0, step / sd_warmup)

    def ema_update(e, p):
        return jax.tree.map(lambda a, b: ema_decay * a + (1 - ema_decay) * b, e, p)

    timer = 0.0
    for step in range(alg_cfg.iters):
        rng, sub = jax.random.split(rng)
        t0 = time()

        (loss_diff, (rng, paths)), g_diff = grad_sampling(
            sub, model_state, model_state.params
        )
        model_state = model_state.apply_gradients(grads=g_diff)
        ema_params = ema_update(ema_params, model_state.params)

        (loss_sd, rng), g_sd = grad_sd(
            rng, model_state, model_state.params, ema_params, paths
        )
        alpha = sd_weight(step)
        g_sd = jax.tree.map(lambda g: alpha * g, g_sd)
        model_state = model_state.apply_gradients(grads=g_sd)
        ema_params = ema_update(ema_params, model_state.params)

        timer += time() - t0

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            rng, sub = jax.random.split(rng)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * batch_size)
            logger["KL/neg_elbo"].append(loss_diff)

            diff_norm = jnp.sqrt(
                sum(jnp.vdot(p, p) for p in jtu.tree_leaves(g_diff))
            )
            sd_norm = jnp.sqrt(
                sum(jnp.vdot(p, p) for p in jtu.tree_leaves(g_sd))
            )

            logger["train/grad_diff"].append(diff_norm)
            logger["train/sd"].append(loss_sd)
            logger["train/grad_sd"].append(sd_norm)

            logger.update(eval_fn(model_state, sub))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
            
            # Checkpoint  
            if step > 0 and checkpoint_managers:
                rng, sub = jax.random.split(rng)
                pkg = dict(
                    model_state=jax.device_get(model_state),
                    key_gen=sub,
                    step=step,
                    timer=timer,
                )

                for name, manager in checkpoint_managers.items():
                    future = manager.save(step, pkg)
                    print(f"[Orbax] step={step} saved to '{name}' (queued={future})")
