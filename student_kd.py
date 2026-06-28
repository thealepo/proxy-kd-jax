import jax
import jax.numpy as jnp
from flax import nnx
from proxy_alignment import get_token_log_probs

def student_nll_loss(student_model , input_ids , teacher_response):
    token_log_probs = get_token_log_probs(student_model , input_ids , teacher_response)
    return -jnp.mean(token_log_probs.sum(-1))