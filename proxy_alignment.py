# dataset: x (input_ids)
import jax
import jax.numpy as jnp
from flax import nnx

def generate(model , input_ids , rng):
    # wnat to return the probability distribution and hard output
    # NOTE: must replace black box model with a true API (irl, we wont get a prob distribution form a black box model)

    logits = model(input_ids)  # [batch , seq_len , vocab_size]

    # log probs
    log_probs = jax.nn.log_softmax(logits , axis=-1)
    # token
    next_token = jax.random.categorical(
        rng , logits[: , -1 , :] , axis=1
    )

    return next_token , log_probs


def preference_loss(teacher_model , proxy_model , input_ids , y_winner , y_loser):
    ...

def proxy_alignment(teacher_model , proxy_model , input_ids , rng):
    pass