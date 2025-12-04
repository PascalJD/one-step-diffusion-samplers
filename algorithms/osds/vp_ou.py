from __future__ import annotations
import jax
import jax.numpy as jnp


def _gaussian_log_prob_diag(
    x: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray
) -> jnp.ndarray:
    d = x.shape[-1]
    inv_var = 1.0 / var
    quad = jnp.sum((x - mean) ** 2, axis=-1) * inv_var
    return -0.5 * (d * jnp.log(2.0 * jnp.pi * var) + quad)


def vp_linear_shrink(
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    beta_min: float,
    beta_max: float
) -> jnp.ndarray:
    d_beta = float(beta_max) - float(beta_min)
    integ = float(beta_min) * (t1 - t0) + 0.5 * d_beta * (t1 * t1 - t0 * t0)
    return jnp.exp(-0.5 * integ)


def vp_linear_var(
    t0: float, t1: float, sigma0: float, beta_min: float, beta_max: float
) -> jnp.ndarray:
    s = vp_linear_shrink(t0, t1, beta_min, beta_max)
    return jnp.maximum(1.0 - s * s, 1e-20) * (sigma0 ** 2)


def vp_ou_backward_logprob_linear(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    sigma0: float,
    beta_min: float,
    beta_max: float
) -> jnp.ndarray:
    s = vp_linear_shrink(t0, t1, beta_min, beta_max)
    var = vp_linear_var(t0, t1, sigma0, beta_min, beta_max)
    mean = s * x1
    return _gaussian_log_prob_diag(x0, mean, var)


def _integrate_beta_trapezoid(
    beta_fn,
    total_steps: int,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    n: int = 4097,
    *,
    dtype=jnp.float32,
) -> jnp.ndarray:
    grid = jnp.linspace(jnp.array(0.0, dtype=dtype), jnp.array(1.0, dtype=dtype), n)

    t0 = jnp.asarray(t0, dtype=dtype)
    t1 = jnp.asarray(t1, dtype=dtype)
    ts = t0[..., None] + (t1 - t0)[..., None] * grid 

    steps = ts * jnp.array(float(total_steps), dtype=dtype)

    steps_flat = steps.reshape((-1,))
    betas_flat = jax.vmap(beta_fn)(steps_flat.astype(dtype))
    betas = betas_flat.reshape(steps.shape)

    dx = ts[..., 1:] - ts[..., :-1] 
    avg = 0.5 * (betas[..., :-1] + betas[..., 1:])
    integral = jnp.sum(avg * dx, axis=-1)
    return integral


def vp_shrink_from_schedule(
    beta_fn,
    total_steps: int,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    n_trapz: int = 4097
) -> jnp.ndarray:
    integ = _integrate_beta_trapezoid(beta_fn, total_steps, t0, t1, n_trapz)
    return jnp.exp(-0.5 * integ)


def vp_var_from_schedule(
    beta_fn, total_steps: int, sigma0: float, t0: float, t1: float
) -> jnp.ndarray:
    s = vp_shrink_from_schedule(beta_fn, total_steps, t0, t1)
    return jnp.maximum(1.0 - s * s, 1e-20) * (sigma0 ** 2)


def vp_ou_backward_logprob_from_schedule(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    beta_fn,
    total_steps: int,
    sigma0: float,
    t0: float = 0.0,
    t1: float = 1.0,
    n_trapz: int = 4097
) -> jnp.ndarray:
    integ = _integrate_beta_trapezoid(beta_fn, total_steps, t0, t1, n_trapz)
    s = jnp.exp(-0.5 * integ)
    var = jnp.maximum(1.0 - s * s, 1e-20) * (sigma0 ** 2)
    mean = s * x1
    return _gaussian_log_prob_diag(x0, mean, var)


def make_ou_weight_fn(
    *,
    prior_std: float,
    noise_schedule,
    schedule_type: str,
    num_steps: int,
    beta_min: float | None = None,
    beta_max: float | None = None,
    n_trapz: int = 257
):
    T = float(num_steps)

    if schedule_type.lower() == "linear":
        compute_s = lambda t0, t1: vp_linear_shrink(
            t0, t1, float(beta_min), float(beta_max)
        )
    else:
        # e.g., "cosine", use numerical integration
        compute_s = lambda t0, t1: vp_shrink_from_schedule(
            noise_schedule, num_steps, t0, t1, n_trapz
        )

    def weight_fn(model_state, params, paths, log_w_em):
        B, Kp1, D = paths.shape
        k = Kp1 - 1
        if (num_steps % k) != 0:
            raise ValueError(f"num_steps={num_steps} must be divisible by k={k}.")
        d  = num_steps // k
        dt = float(d) / T

        # Training scan uses descending codes [T, T-d, ..., d]
        codes_desc = jnp.arange(d, num_steps + 1, d, dtype=jnp.int32)[::-1]
        codes_f32  = codes_desc.astype(jnp.float32)
        t1 = codes_f32 / T
        t0 = t1 - dt

        # Path slices per step
        Xn   = paths[:, :-1, :]
        Xnp1 = paths[:,  1:, :]

        # EM backward kernel (discrete step beta)
        beta_t = jax.vmap(lambda c: noise_schedule(c))(codes_f32)
        var_em  = beta_t * (prior_std ** 2) * dt
        mean_em = Xnp1 - 0.5 * (beta_t * dt)[None, :, None] * Xnp1 
        log_bwd_em = _gaussian_log_prob_diag(Xn, mean_em, var_em[None, :])

        # OU backward params per step (exact in continuous-time)
        s_vec  = compute_s(t0, t1)
        var_ou = (1.0 - s_vec * s_vec) * (prior_std ** 2)
        mean_ou = s_vec[None, :, None] * Xnp1
        log_bwd_ou = _gaussian_log_prob_diag(Xn, mean_ou, var_ou[None, :])

        # Weight correction
        delta    = jnp.sum(log_bwd_ou - log_bwd_em, axis=1)
        log_w_ou = log_w_em + delta
        return log_w_ou

    return weight_fn