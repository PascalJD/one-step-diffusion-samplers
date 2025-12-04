import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial


def sample_d_and_t(rng, num_steps: int):
    n = jnp.asarray(num_steps, dtype=jnp.int32)
    rng_d, rng_t = jax.random.split(rng)

    k_max = jnp.int32(jnp.log2(n))
    k = jax.random.randint(rng_d, (), 1, k_max + 1)
    d = jnp.int32(1 << k)

    choices = n // d
    m = jax.random.randint(rng_t, (), 0, choices)
    t = m * d
    return d, t


def distillation_loss(
    rng,
    model_state,
    params,
    teacher_params,
    paths,
    batch_size,
    prior_tuple,
    target,
    betas,
    num_steps,
    *,
    trace_weight: float = 0.1,
    n_trace_probes: int = 1,
    jac_weight: float = 1e-5,
):
    rngs = jax.random.split(rng, num=batch_size)

    _psm_bound = partial(
        _per_sample_mse,
        trace_weight=trace_weight,
        n_trace_probes=n_trace_probes,
        jac_weight=jac_weight,
    )

    mse = jax.vmap(
        _psm_bound,
        in_axes=(0, None, None, None, 0, None, None, None, None),
    )(
        rngs,
        model_state,
        params,
        teacher_params,
        paths,
        prior_tuple,
        target,
        betas,
        num_steps,
    )
    return jnp.mean(mse)


def _per_sample_mse(
    rng,
    model_state,
    params,
    teacher_params,
    paths,
    prior_tuple,
    target,
    betas,
    num_steps,
    *,
    trace_weight: float = 0.1,
    n_trace_probes: int = 1,
    jac_weight: float = 1e-5,
):
    prior_std, prior_sample, prior_logp = prior_tuple
    target_logp = target.log_prob
    paths = jax.lax.stop_gradient(paths)

    def lgv_init(x, t_code, sigma_t, t_total):
        tr = t_code / t_total
        return sigma_t * (
            (1.0 - tr) * target_logp(x) + tr * prior_logp(x)
        )
    lgv_init = partial(lgv_init, t_total=num_steps)

    def _hutch_trace(u_fn, x, key, n=1):
        keys = jax.random.split(key, n)
        def one(k):
            v = jnp.sign(jax.random.normal(k, shape=x.shape))
            return jnp.vdot(v, jax.jvp(u_fn, (x,), (v,))[1])
        return jnp.mean(jax.vmap(one)(keys))

    def pf_ode_step(state, t_code, d_code, params_local, key_trace):
        x, rng_local = state

        beta_t = betas(t_code)
        sigma_t = jnp.sqrt(beta_t) * prior_std

        lgv = jax.lax.stop_gradient(
            jax.grad(lgv_init)(x, t_code, sigma_t)
        )

        t_norm = (t_code / num_steps) * jnp.ones(1)
        d_norm = (d_code / num_steps) * jnp.ones(1)

        def u_fn(z):
            return model_state.apply_fn(params_local, z, t_norm, d_norm, lgv)
        u = u_fn(x)

        dt = d_code / num_steps
        x_new = x + (0.5 * sigma_t * u + 0.5 * beta_t * x) * dt

        dim = x.shape[0]
        tr_du = _hutch_trace(u_fn, x, key_trace, n=n_trace_probes)
        div_b = 0.5 * beta_t * dim + 0.5 * sigma_t * tr_du
        logdet_inc = dt * div_b

        return (x_new, rng_local), logdet_inc

    d_code, t_idx = sample_d_and_t(rng, num_steps)
    d_half = d_code >> 1
    t_half_idx = t_idx + d_half
    t_code = num_steps - t_idx
    t_half_code = num_steps - t_half_idx

    x_t = paths[t_idx, :]
    state0 = (x_t, rng)

    # Teacher two half steps
    params_sg = jtu.tree_map(jax.lax.stop_gradient, teacher_params)
    rng, k1, k2, kjac = jax.random.split(rng, 4)

    (x_mid, _), logdet_half1 = pf_ode_step(state0, t_code, d_half, params_sg, k1)
    (x_tgt, _), logdet_half2 = pf_ode_step((x_mid, rng), t_half_code, d_half, params_sg, k2)
    logdet_two = jax.lax.stop_gradient(logdet_half1 + logdet_half2)

    # Student one big step
    (x_pred, _), logdet_big = pf_ode_step(state0, t_code, d_code, params, k1)

    # Loss components
    L_state = jnp.mean((x_tgt - x_pred) ** 2)
    L_trace = (logdet_two - logdet_big) ** 2

    # Jacobian penalty at the starting point to stabilize Lipschitz
    def u_fn_student(z):
        t_norm = (t_code / num_steps) * jnp.ones(1)
        d_norm = (d_code / num_steps) * jnp.ones(1)
        beta_t = betas(t_code)
        sigma_t = jnp.sqrt(beta_t) * prior_std
        lgv = jax.lax.stop_gradient(jax.grad(lgv_init)(x_t, t_code, sigma_t))
        return model_state.apply_fn(params, z, t_norm, d_norm, lgv)

    vjac = jnp.sign(jax.random.normal(kjac, shape=x_t.shape))
    Jv = jax.jvp(u_fn_student, (x_t,), (vjac,))[1]
    L_jac = jnp.mean(Jv ** 2)

    return L_state + trace_weight * L_trace + jac_weight * L_jac