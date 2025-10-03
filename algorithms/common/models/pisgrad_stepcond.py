import jax.numpy as jnp
from flax import linen as nn

class PISGRADStep(nn.Module):
    dim: int
    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2
    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        # Frequencies for Fourier features
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ])

        layers_grad = []
        layers_grad.append(nn.Dense(self.num_hid))
        for _ in range(self.num_layers):
            layers_grad.extend([nn.gelu, nn.Dense(self.num_hid)])
        layers_grad.append(
            nn.Dense(
                self.dim,
                kernel_init=nn.initializers.constant(self.weight_init),
                bias_init=nn.initializers.constant(self.bias_init)
            )
        )
        self.time_coder_grad = nn.Sequential(layers_grad)

        layers_state_time = []
        for _ in range(self.num_layers):
            layers_state_time.append(nn.Sequential([nn.Dense(self.num_hid), nn.gelu]))
        layers_state_time.append(
            nn.Dense(
                self.dim,
                kernel_init=nn.initializers.constant(1e-8),
                bias_init=nn.initializers.zeros_init()
            )
        )
        self.state_time_net = nn.Sequential(layers_state_time)

    def get_fourier_features(self, timesteps):
        # Expand dims if needed so that shape is (batch, 1).
        if timesteps.ndim == 0:
            timesteps = timesteps[None, None]  # e.g. shape (1,1)
        elif timesteps.ndim == 1:
            timesteps = timesteps[:, None])

        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, x_array, t_array, d_array, lgv_term):
        # 0) Handle singleton batch
        was_single = (x_array.ndim == 1)
        if was_single:
            # expand to (1, dim) if needed
            x_array = x_array[None, :]
            lgv_term = lgv_term[None, :]

        # 1) Compute Fourier features for t and for d
        t_emb = self.get_fourier_features(t_array)  # shape (batch, 2*num_hid)
        d_emb = self.get_fourier_features(d_array)  # same shape
        td_emb = jnp.concatenate([t_emb, d_emb], axis=-1)  # shape (batch, 4*num_hid)

        # 3) Pass through two subnets: time_coder_state and time_coder_grad
        t_net1 = self.time_coder_state(td_emb)  # shape (batch, num_hid)
        t_net2 = self.time_coder_grad(td_emb)  # shape (batch, dim)

        # 4) Combine with x_array: we append t_net1 to x_array
        extended_input = jnp.concatenate([x_array, t_net1], axis=-1)
        # shape = (batch, dim + num_hid)

        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        # 5) Clip lgv_term and combine
        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * lgv_term  # shape (batch, dim)

        if was_single:
            out_state_p_grad = out_state_p_grad[0]
        
        return out_state_p_grad