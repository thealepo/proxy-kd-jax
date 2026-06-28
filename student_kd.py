import jax
import jax.numpy as jnp
from flax import nnx
from proxy_alignment import autoregressive_generation, get_token_log_probs

ALPHA = ...  # NOTE: ADD LATER (pay attn to paper)

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
        mean = jnp.mean(log_probs_proxy , axis=0 , keepdims=True)
        std = jnp.std(log_probs_proxy , axis=0 , keepdims=True)

        # Inner
        logits = (log_probs_proxy - mean) / (std + 1e-8)
        return jax.nn.sigmoid(logits)

    # Logits
    weight = jax.lax.stop_gradient(kl_weight(log_probs_proxy)) # this is a fixed taget with no grad
    weighted_kl_loss = weight * ratio

    return jnp.mean(weighted_kl_loss.sum(-1))

@nnx.jit
def train_step(proxy_model , student_model , optimizer , batch):
    input_ids , teacher_response , proxy_response = batch

    def loss_fn(student_model):
        nll_loss = student_nll_loss(student_model , input_ids , teacher_response)
        weighted_kl_loss = student_kl_loss(proxy_model , proxy_response , student_model , input_ids)
        return nll_loss + ALPHA * weighted_kl_loss

    # Updates & Autograd
    loss , grads = nnx.value_and_grad(loss_fn)(student_model)
    optimizer.update(student_model , grads)
    return loss

# DATA COLLECTION STEP
# THE PROXY IS ALREADY ALIGNED AT THIS POINT
def collection(teacher_model , proxy_model , input_ids , rng , max_new_tokens=256):
    # RNG sutff
    rng , rng_teacher , rng_proxy = jax.random.split(rng , 3)

    # Responses
    teacher_response = teacher_model.generate(input_ids , rng_teacher)  # IRL THIS IS AN API CALL
    proxy_full = autoregressive_generation(proxy_model , input_ids , rng_proxy , max_new_tokens=max_new_tokens)
    proxy_response = proxy_full[: , input_ids.shape[1]:]

    return input_ids , teacher_response , proxy_response