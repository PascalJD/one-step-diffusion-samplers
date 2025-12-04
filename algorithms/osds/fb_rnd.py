import jax
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as npdist

def sample_kernel(rng, mean, scale):
    eps = jax.random.normal(rng, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def _make_step_codes(
        total_steps: int, eval_steps: int | None
    ) -> jnp.ndarray:
    t = int(total_steps)
    if eval_steps is None:
        k = t
    else:
        k = int(eval_steps)
        if k > t:
            raise ValueError(f"eval_steps={k} > total_steps={t}.")
        if (t % k) != 0:
            raise ValueError(f"T={t} must be divisible by k={k}.")
    d = t // k
    return jnp.arange(d, t + 1, d, dtype=jnp.int32)


def per_sample_rnd(
    rng,
    model_state,
    params,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
    prior_to_target=True,
    use_ode=True,
    *,
    schedule_type: str = "linear",
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 1025,
    integrator_kind: str = "em", 
):
    prior_std, prior_sample, prior_logp = prior_tuple
    target_logp = target.log_prob

    step_codes = _make_step_codes(num_steps, eval_steps)
    k = int(step_codes.shape[0])
    t_float = float(num_steps)
    d = t_float / float(k)
    dt = d / t_float

    def lgv_init(x, t_code, sigma_t, t_total):
        tr = t_code / t_total
        return sigma_t * (
            (1.0 - tr) * target_logp(x) + tr * prior_logp(x)
        )
    lgv_init = partial(lgv_init, t_total=t_float)

    def _beta_sigma_for_pf(sp):
        if integrator_kind == "midpoint_em":
            return sp.bar_beta, sp.sigma_bar
        return sp.beta_t1, sp.sigma_t1

    def _div_u_hutch(x, t_ctrl, d_norm, lgv, rng_key):
        """
        Hutchinson estimator of tr(du/dx) at (x, t_ctrl, d_norm, lgv).
        """
        def u_fn(z):
            return model_state.apply_fn(
                params,
                z,
                t_ctrl * jnp.ones(1),
                d_norm * jnp.ones(1),
                lgv
            )
        # Rademacher via sign of normal
        v = jnp.sign(jax.random.normal(rng_key, shape=x.shape))
        jvp_val = jax.jvp(u_fn, (x,), (v,))[1]  # = J_u(x) @ v
        return jnp.vdot(v, jvp_val)  # v^T J v  -> unbiased for tr(J)

    def sim_prior_to_target(state, t_code):
        # state carry
        x, log_w_em, logdet_ode, rng_inner = state
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # per-step parameters
        t_code = t_code.astype(jnp.float32)
        sp = compute_step_params(
            kind=integrator_kind,
            t1_code=t_code, 
            dt=dt, 
            prior_std=prior_std,
            noise_schedule=noise_schedule, 
            num_steps=num_steps,
            schedule_type=schedule_type, 
            beta_min=beta_min,
            beta_max=beta_max, 
            n_trapz=n_trapz
        )

        # control at integrator-defined time 
        lgv_sigma = sigma_for_lgv(integrator_kind, sp)
        lgv = jax.lax.stop_gradient(jax.grad(lgv_init)(x, t_code, lgv_sigma))
        t_ctrl = control_time(integrator_kind, sp)
        u = model_state.apply_fn(
            params, 
            x,
            t_ctrl * jnp.ones(1),
            (d / t_float) * jnp.ones(1),
            lgv
        )

        # forward kernel
        fwd_m = integ_fwd_mean(integrator_kind, x, u, sp)
        sc = integ_scale(integrator_kind, sp)

        if use_ode:
            # PF ODE shortcuts for sampling
            x_new = ode_drift(integrator_kind, x, u, sp)

            # accumulate log|det nabla T| via divergence integral
            beta_pf, sigma_pf = _beta_sigma_for_pf(sp)
            dim = x.shape[0]
            div_lin = 0.5 * beta_pf * dim
            rng_inner, key_h = jax.random.split(rng_inner)
            tr_du = _div_u_hutch(x, t_ctrl, d / t_float, lgv, key_h)
            div_ctrl = 0.5 * sigma_pf * tr_du

            div_b = div_lin + div_ctrl
            logdet_ode = logdet_ode + sp.dt * div_b
        else:
            rng_inner, sub = jax.random.split(rng_inner)
            x_new = sample_kernel(sub, fwd_m, sc)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # backward kernel
        bwd_m = integ_bwd_mean(integrator_kind, x_new, sp)

        # log-likelihoods
        fwd_lp = log_prob_kernel(x_new, fwd_m, sc)
        bwd_em_lp = log_prob_kernel(x, bwd_m, sc)
        log_w_em = log_w_em + (bwd_em_lp - fwd_lp)

        return (x_new, log_w_em, logdet_ode, rng_inner), x_new

    def sim_target_to_prior(state, t_code):
        x, log_w_em, logdet_ode, rng_inner = state
        t_code = t_code.astype(jnp.float32)

        # per-step parameters
        sp = compute_step_params(
            kind=integrator_kind,
            t1_code=t_code, 
            dt=dt, 
            prior_std=prior_std,
            noise_schedule=noise_schedule, 
            num_steps=num_steps,
            schedule_type=schedule_type, 
            beta_min=beta_min,
            beta_max=beta_max, 
            n_trapz=n_trapz
        )
        sc = integ_scale(integrator_kind, sp)

        if use_ode:
            # Inverse PF-ODE step: move toward t=0 with the negative drift,
            lgv_sigma = sigma_for_lgv(integrator_kind, sp)
            lgv = jax.lax.stop_gradient(jax.grad(lgv_init)(x, t_code, lgv_sigma))
            t_ctrl = control_time(integrator_kind, sp)
            u = model_state.apply_fn(
                params, 
                x, 
                t_ctrl * jnp.ones(1), 
                (d / t_float) * jnp.ones(1), 
                lgv
            )

            # One Euler step of the inverse map
            beta_pf, sigma_pf = _beta_sigma_for_pf(sp)
            g = (0.5 * sigma_pf) * u + (0.5 * beta_pf) * x
            x_new = x - sp.dt * g

            # accumulate log |det dT| just like in the forward branch
            dim = x.shape[0]
            div_lin  = 0.5 * beta_pf * dim
            rng_inner, key_h = jax.random.split(rng_inner)
            tr_du    = _div_u_hutch(x, t_ctrl, d / t_float, lgv, key_h)
            div_ctrl = 0.5 * sigma_pf * tr_du
            logdet_ode = logdet_ode + sp.dt * (div_lin + div_ctrl)

            if stop_grad:
                x_new = jax.lax.stop_gradient(x_new)

            # We keep EM kernels around for consistency bookkeeping, but they won't be used
            bwd_m = integ_bwd_mean(integrator_kind, x, sp)
            fwd_m = integ_fwd_mean(integrator_kind, x, u, sp)
            fwd_lp = log_prob_kernel(x_new, fwd_m, integ_scale(integrator_kind, sp))
            bwd_lp = log_prob_kernel(x, bwd_m, integ_scale(integrator_kind, sp))
            log_w_em = log_w_em + (bwd_lp - fwd_lp)
            return (x_new, log_w_em, logdet_ode, rng_inner), x_new

        # sample from backward kernel (denominator path)
        bwd_m = integ_bwd_mean(integrator_kind, x, sp)
        rng_inner, sub = jax.random.split(rng_inner)
        x_new = sample_kernel(sub, bwd_m, sc)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # control at right endpoint state, but integrator-defined time
        lgv_sigma = sigma_for_lgv(integrator_kind, sp)
        lgv = jax.lax.stop_gradient(jax.grad(lgv_init)(x_new, t_code, lgv_sigma))
        t_ctrl = control_time(integrator_kind, sp)
        u = model_state.apply_fn(
            params, 
            x_new,
            t_ctrl * jnp.ones(1),
            (d / t_float) * jnp.ones(1),
            lgv
        )

        # forward kernel scored at right endpoint state
        fwd_m = integ_fwd_mean(integrator_kind, x_new, u, sp)

        fwd_lp = log_prob_kernel(x, fwd_m, sc)
        bwd_lp = log_prob_kernel(x_new, bwd_m, sc)
        log_w_em = log_w_em + (bwd_lp - fwd_lp)
        return (x_new, log_w_em, logdet_ode, rng_inner), x_new

    rng, sub = jax.random.split(rng)

    if prior_to_target:
        codes_desc = step_codes[::-1]
        x0 = jnp.clip(prior_sample(seed=sub), -4 * prior_std, 4 * prior_std)
        state0 = (x0, 0.0, 0.0, rng)
        (xT, log_ratio_em, logdet_ode, _), xs = jax.lax.scan(
            sim_prior_to_target, state0, codes_desc
        )
        term_c = prior_logp(x0) - target_logp(xT)
    else:
        codes_asc = step_codes
        x0 = target.sample(sub, ())
        state0 = (x0, 0.0, 0.0, rng)
        (xT, log_ratio_em, logdet_ode, _), xs= jax.lax.scan(
            sim_target_to_prior, state0, codes_asc
        )
        term_c = prior_logp(xT) - target_logp(x0)

    traj = jnp.concatenate([x0[None, ...], xs], axis=0)
    stoch_c = logdet_ode if use_ode else jnp.zeros_like(log_ratio_em)
    run_c_em = -log_ratio_em

    return xT, run_c_em, stoch_c, term_c, traj


def rnd(
    rng,
    model_state,
    params,
    batch_size,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
    prior_to_target=True,
    use_ode=False,
    return_traj=False,
    *,
    schedule_type: str = "linear",
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 1025,
    integrator_kind: str = "em", 
):
    rngs = jax.random.split(rng, num=batch_size)
    def _one(r):
        return per_sample_rnd(
            r, 
            model_state, 
            params, 
            prior_tuple, 
            target,
            num_steps, 
            noise_schedule,
            eval_steps,
            stop_grad,
            prior_to_target,
            use_ode,
            schedule_type=schedule_type,
            beta_min=beta_min,
            beta_max=beta_max,
            n_trapz=n_trapz,
            integrator_kind=integrator_kind,
        )
    xT, rc_em, sc, tc, traj = jax.vmap(_one)(rngs)
    if return_traj:
        return xT, rc_em, sc, tc, traj
    return xT, rc_em, sc, tc


def neg_elbo(
    rng,
    model_state,
    params,
    batch_size,
    prior_tuple,
    target,
    num_steps,
    noise_schedule,
    eval_steps=None,
    stop_grad=False,
):
    xT, rc, _, tc = rnd(
        rng,
        model_state,
        params,
        batch_size,
        prior_tuple,
        target,
        num_steps,
        noise_schedule,
        eval_steps=eval_steps,
        stop_grad=stop_grad,
        prior_to_target=True,
        use_ode=False,
        return_traj=False,
    )
    val = rc + tc
    return jnp.mean(val), (val, xT)