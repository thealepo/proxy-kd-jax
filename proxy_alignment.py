# dataset: x (input_ids)
import jax
import jax.numpy as jnp
from flax import nnx

def generation(model , input_ids , rng):
    ...

def preference_loss(input_ids , y_winner , y_loser):
    pass

def proxy_alignment(teacher_model , proxy_model , input_ids , rng):
    pass