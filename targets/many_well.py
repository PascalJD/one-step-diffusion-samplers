import math
from typing import List

import chex
import distrax
import jax
import jax.numpy as jnp
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb
from jax import random
from targets.base_target import Target
from utils.plot import plot_contours_2D, plot_marginal_pair


# Taken from FAB code
class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1.0):
        dim = 2
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1


class ManyWellEnergy(Target):
    def __init__(
        self,
        a: float = -0.5,
        b: float = -6.0,
        c: float = 1.0,
        dim=32,
        can_sample=False,
        sample_bounds=None,
    ) -> None:
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(a, b, c)

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        self.centre = 1.7
        self.max_dim_for_all_modes = (
            40  # otherwise we get memory issues on huge test set
        )
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(
                *[jnp.array([-self.centre, self.centre]) for _ in range(self.n_wells)]
            )
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.0

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_probs = jnp.sum(
            jnp.stack(
                [
                    self.double_well_energy.log_prob(x[..., i * 2 : i * 2 + 2])
                    for i in range(self.n_wells)
                ],
                axis=-1,
            ),
            axis=-1,
            keepdims=True,
        ).reshape((-1,))

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)
        return log_probs

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def visualise(
        self, samples: chex.Array, axes: List[plt.Axes] = None, show: bool = False
    ) -> None:
        """Visualise samples from the model."""
        plt.close()
        fig, ax = plt.subplots()

        plot_contours_2D(self.log_prob_2D, ax, bound=self._plot_bound, levels=20)
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None


class ManyWell2(Target):
    def __init__(
        self,
        dim: float = 5,
        m: float = 5,
        delta: float = 4,
        can_sample: bool = False,
        sample_bounds=None,
    ):
        self.d = dim
        self.m = m
        self.delta = jnp.array(delta)

        self._plot_bound = 3.0

        super().__init__(dim=dim, log_Z=self.log_Z, can_sample=can_sample)

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]
        assert x.shape[1] == self.d, "Dimension mismatch"
        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :m]
        k = ((prefix**2 - delta) ** 2).sum(axis=1)

        suffix = x[:, m:]
        k2 = 0.5 * (suffix**2).sum(axis=1)

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def log_prob_2D(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :2]
        k = ((prefix**2 - delta) ** 2).sum(axis=1)

        # suffix = x[:, m:]
        # k2 = 0.5 * (suffix**2).sum(axis=1)
        k2 = 0

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(
        self,
        samples: chex.Array,
        axes: List[plt.Axes] = None,
        show: bool = False,
        savefig: bool = False,
    ) -> None:
        """Visualise samples from the model."""
        plt.close()
        fig, ax = plt.subplots()

        plot_contours_2D(self.log_prob_2D, ax, bound=self._plot_bound, levels=20)
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        if savefig:
            plt.savefig("vis.png")
        return wb

    @property
    def log_Z(self):
        # numerical integration
        l, r = -100, 100
        s = 100000000
        key = jax.random.PRNGKey(0)

        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        fst = jnp.log(jnp.sum(jnp.exp(-((pt**2 - self.delta) ** 2)) * ((r - l) / s)))

        self.logZ_1d = fst

        # well the below works but there's analytic solution this is Gaussian lmao - junhua
        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        snd = jnp.log(jnp.sum(jnp.exp(-0.5 * pt**2) * ((r - l) / s)))

        return fst * self.m + snd * (self.d - self.m)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        """JIT-safe rejection sampling for the Many-Well target.

        Returns samples with shape: sample_shape + (self.d,)
        First self.m coordinates are from the 1D double-well; the rest are Gaussian.
        """
        REJECTION_SCALE = 6.0  # proposal/target envelope (same intent as before)

        # 1D double-well log-density (normalized via self.logZ_1d)
        def doubleWell1dLogDensity(xs, shift, separation):
            # xs shape (..., 1)
            return -(((xs - shift) ** 2 - separation) ** 2) - self.logZ_1d

        def GetProposal(shift, separation):
            # proposal distribution for 1D double-well rejection sampling
            loc = shift + jnp.sqrt(separation) * jnp.array([[-1.0], [1.0]])
            scale = 1.0 / jnp.sqrt(separation) * jnp.array([[1.0], [1.0]])
            ps = jnp.array([0.5, 0.5])
            components = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            return distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(probs=ps),
                components_distribution=components,
            )

        def rejection_sampling_fixed(
            key, shape, proposal, log_target, scale: float, batch_size: int = 1024
        ):
            """
            JIT-safe rejection sampler that returns exactly prod(shape) samples
            with event shape (1,), reshaped to shape + (1,) at the end.
            """
            total = int(math.prod(shape))  # static py int is fine in closures

            # Buffer for accepted samples: (total, 1)
            buf = jnp.zeros((total, 1))
            count0 = jnp.array(0, dtype=jnp.int32)

            def cond_fun(carry):
                key_c, buf_c, count_c = carry
                return count_c < total

            def body_fun(carry):
                key_c, buf_c, count_c = carry
                key_c, k1, k2 = random.split(key_c, 3)

                # Draw a fixed-size batch of candidates: (B, 1)
                xs = proposal.sample(seed=k1, sample_shape=(batch_size,))
                log_p = proposal.log_prob(xs)                 # (B,)
                log_t = log_target(xs).reshape((batch_size,)) # (B,)
                # Accept with prob = exp(log_t - log_p - log(scale)) <= 1
                log_acc = log_t - log_p - jnp.log(scale)
                probs = jnp.exp(jnp.minimum(log_acc, 0.0))
                u = random.uniform(k2, (batch_size,))
                accept = u < probs                            # (B,)

                # We will write up to `remaining` accepted samples from this batch
                remaining = jnp.minimum(
                    jnp.array(total, dtype=count_c.dtype) - count_c,
                    jnp.array(batch_size, dtype=count_c.dtype),
                )

                def write_one(i, state):
                    b, taken = state
                    cond_i = jnp.logical_and(accept[i], taken < remaining)
                    idx = count_c + taken

                    def do_write(bb):
                        return bb.at[idx, 0].set(xs[i, 0])

                    b = jax.lax.cond(cond_i, do_write, lambda bb: bb, b)
                    taken = jax.lax.select(cond_i, taken + 1, taken)
                    return (b, taken)

                b_new, taken = jax.lax.fori_loop(
                    0, batch_size, write_one, (buf_c, jnp.array(0, dtype=count_c.dtype))
                )
                return (key_c, b_new, count_c + taken)

            key_out, buf_out, count_out = jax.lax.while_loop(
                cond_fun, body_fun, (key, buf, count0)
            )
            # buf_out has shape (total, 1) -> reshape to desired output
            return buf_out.reshape(shape + (1,))

        # ---- assemble the full d-dimensional sample ----
        key, subkey1, subkey2 = random.split(seed, num=3)
        n_dw = int(self.m)
        n_gauss = int(self.d - self.m)

        proposal = GetProposal(0.0, self.delta)
        log_target = lambda xs: doubleWell1dLogDensity(xs, 0.0, self.delta)

        # Double-well dims: sample_shape + (n_dw,)
        dw_samples = rejection_sampling_fixed(
            subkey1, sample_shape + (n_dw,), proposal, log_target, REJECTION_SCALE
        ).squeeze(-1)  # drop the event dim -> (..., n_dw)

        # Remaining dims: standard Gaussian
        gauss_samples = random.normal(subkey2, sample_shape + (n_gauss,))

        # Concatenate along features
        return jnp.concatenate([dw_samples, gauss_samples], axis=-1)


if __name__ == "__main__":
    # mw = ManyWellEnergy()
    # mw.visualise(samples=mw.sample(jax.random.PRNGKey(0), (1,)))

    key = jax.random.PRNGKey(42)
    well = ManyWellEnergy()

    samples = jax.random.normal(key, shape=(10, 32))
    print(samples.shape)
    print((well.log_prob(samples)))
    print((jax.vmap(well.log_prob)(samples)))

    mwb = ManyWell2(dim=5, m=5, delta=4)

    samples = jax.random.normal(key, shape=(10, 5))
    print(mwb.log_prob(samples))
    print(mwb.log_Z)

    mwb.visualise(samples=jax.random.normal(key, shape=(1000, 50)), savefig=True)
