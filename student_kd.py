import jax
import jax.numpy as jnp
from flax import nnx
from proxy_alignment import get_token_log_probs

ALPHA = ...

def student_nll_loss(student_model , input_ids , teacher_response):
    token_log_probs = get_token_log_probs(student_model , input_ids , teacher_response)
    return -jnp.mean(token_log_probs.sum(-1))

def student_kl_loss(proxy_model , proxy_response , student_model , input_ids):
    # gather log probs
    log_probs_proxy = get_token_log_probs(proxy_model , input_ids , proxy_response)
    log_probs_student = get_token_log_probs(student_model , input_ids , proxy_response)

    # Ratio
    ratio = log_probs_proxy - log_probs_student

    # Weihgt
    def kl_weight(log_probs_proxy , input_ids , proxy_response):
        # ocmpute mean and std
        mean =
        std = jnp.std(log_probs_proxy , axis=0 , keepdims=True)

    # Logits
    weight = kl_weight(log_probs_proxy , input_ids , proxy_response)
    weighted_kl_loss = weight * ratio

    return -jnp.mean(weighted_kl_loss.sum(-1))