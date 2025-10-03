import optax
from flax.training import train_state

from algorithms.common.models.pisgrad_net import PISGRADNet
from algorithms.common.models.pisgrad_stepcond import PISGRADStep
import jax
import jax.numpy as jnp

from utils.helper import flattened_traversal


def init_model(key, dim, alg_cfg, params=None):
    key, key_gen = jax.random.split(key)
    
    if alg_cfg.name in ["sdss", "sdss_vp"]:
        model = PISGRADStep(**alg_cfg.model)
        if params is None:
            params = model.init(
                key,
                jnp.ones([alg_cfg.batch_size, dim]),
                jnp.ones([alg_cfg.batch_size, 1]),
                jnp.ones([alg_cfg.batch_size, 1]),
                jnp.ones([alg_cfg.batch_size, dim])
            )
        optimizer = optax.chain(
            optax.zero_nans(),
            # optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
            optax.clip_by_global_norm(alg_cfg.grad_clip),
            optax.adamw(learning_rate=alg_cfg.step_size, weight_decay=0.1),
        )
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    model = PISGRADNet(**alg_cfg.model)
    if params is None:
        params = model.init(
            key,
            jnp.ones([alg_cfg.batch_size, dim]),
            jnp.ones([alg_cfg.batch_size, 1]),
            jnp.ones([alg_cfg.batch_size, dim])
        )
    if alg_cfg.name == 'gfn':
        additional_params = {'logZ': jnp.array((alg_cfg.init_logZ,))}
        params['params'] = {**params['params'], **additional_params}

        optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
            optax.masked(optax.adam(learning_rate=alg_cfg.step_size),
                         mask=flattened_traversal(lambda path, _: path[-1] != 'logZ')),
            optax.masked(optax.adam(learning_rate=alg_cfg.logZ_step_size),
                         mask=flattened_traversal(lambda path, _: path[-1] == 'logZ')),
        )
    else:        
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
                                optax.adam(learning_rate=alg_cfg.step_size))

    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
