import jax.numpy as jnp

def get_trajectory_tubes(Phi_x):
    return jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)