import os
from typing import List

import jax.numpy as jnp
import distrax
import chex
import jax.random
import matplotlib.pyplot as plt
import wandb

from targets.base_target import Target


class Funnel(Target):
    def __init__(self, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)
        self.data_ndim = dim
        self.dist_dominant = distrax.Normal(0.0, 3.0)
        self.mean_other = jnp.zeros(dim - 1, dtype=float)
        self.cov_eye = jnp.eye(dim - 1)  # (k, k)
        self.sample_bounds = sample_bounds

    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = jnp.exp(x[:, 0:1])
        neglog_density_other = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, axis=-1)

        log_prob = log_density_dominant + log_density_other
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def sample(self, seed, sample_shape=()):
        key1, key2 = jax.random.split(seed)

        # dominant_x: shape = sample_shape
        dominant_x = self.dist_dominant.sample(seed=key1, sample_shape=sample_shape)
        # x_others: shape = sample_shape + (dim-1,)
        x_others = self._dist_other(dominant_x).sample(seed=key2)

        # make dominant a column: sample_shape + (1,)
        dom_col = dominant_x[..., None]
        out = jnp.concatenate([dom_col, x_others], axis=-1)

        if self.sample_bounds is not None:
            out = out.clip(min=self.sample_bounds[0], max=self.sample_bounds[1])
        return out

    def _dist_other(self, dominant_x):
        # dominant_x shape: sample_shape
        variance_other = jnp.exp(dominant_x)
        cov_other = variance_other[..., None, None] * self.cov_eye  # sample_shape + (k, k)
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)

    def visualise(self, samples: chex.Array = None, axes: List[plt.Axes] = None, show=False, prefix='') -> dict:
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()
        x, y = jnp.meshgrid(jnp.linspace(-10, 5, 100), jnp.linspace(-5, 5, 100))
        grid = jnp.c_[x.ravel(), y.ravel()]
        pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
        pdf_values = jnp.reshape(pdf_values, x.shape)
        plt.contourf(x, y, pdf_values, levels=20, cmap='viridis')
        if samples is not None:
            idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
            ax.scatter(samples[idx, 0], samples[idx, 1], c='r', alpha=0.5, marker='x')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        plt.xticks([])
        plt.yticks([])
        # plt.xlim(-10, 5)
        # plt.ylim(-5, 5)

        # plt.savefig(os.path.join(project_path('./samples/funnel/'), f"{prefix}funnel.pdf"), bbox_inches='tight', pad_inches=0.1)

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb
