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
    def kl_weight(log_probs_proxy):
        # ocmpute mean and std
        mean = jnp.mean(log_probs_proxy , axis=0)
        std = jnp.std(log_probs_proxy , axis=0 , keepdims=True)

        # Inner
        logits = (log_probs_proxy - mean) / std
        return jax.nn.sigmoid(logits)

    # Logits
    weight = kl_weight(log_probs_proxy)
    weighted_kl_loss = weight * ratio

    return -jnp.mean(weighted_kl_loss.sum(-1))

@nnx.jit
def train_step(proxy_model , proxy_response , student_model , optimizer , teacher_response , input_ids):
    def loss_fn(student_model):
        nll_loss = student_nll_loss(student_model , input_ids , teacher_response)
        weighted_kl_loss = student_kl_loss(proxy_model , proxy_response , student_model , input_ids)
        return nll_loss + ALPHA * weighted_kl_loss

    # Updates & Autograd
    loss , grads = nnx.value_and_grad(loss_fn)(student_model)
    optimizer.update(student_model , grads)
    return loss