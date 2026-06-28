import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformer import CausalLanguageModel, TransformerConfig
from utils import get_token_log_probs , collection

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

def train_epoch(teacher_model , proxy_model , student_model , optimizer , prompt_batches , rng , max_new_tokens):
    losses = []

    # loop
    for x in prompt_batches:
        rng , rng_collection = jax.random.split(rng)
        batch = collection(teacher_model , proxy_model , x , rng_collection , max_new_tokens)
        loss = train_step(proxy_model , student_model , optimizer , batch)
        losses.append(float(loss))

    return sum(losses) / len(losses)

def train(teacher_model , proxy_model , student_model , optimizer , prompt_batches , rng , max_new_tokens , num_epochs=3):
    
    for epoch in range(num_epochs):
        rng , rng_epoch = jax.random.split(rng)
        mean_loss = train_epoch(
            teacher_model , proxy_model , student_model , optimizer , prompt_batches , rng , max_new_tokens
        )
        print(f'Epoch; {epoch+1} , avg_loss: {mean_loss:.4f}')

    return student_model

# MOCK RUN (phase ii)
if __name__ == "__main__":
    config = TransformerConfig()
    NUM_BATCHES , BATCH , PROMPT_LEN , MAX_NEW_TOKENS = 3 , 4 , 8 , 8

    # RNG
    rng = jax.random.PRNGKey(42)

    
    # Proxy: pretend its already aligned from phase i: forzen, only sampled from
    proxy_model = CausalLanguageModel(config , rngs=nnx.Rngs(1))
    # Student: the model we distill into (truthfully, smaller than Proxy)
    student_model = CausalLanguageModel(config , rngs=nnx.Rngs(2))
    # Teacher mock
    teacher_transformer = CausalLanguageModel(config , rngs=nnx.Rngs(3))
    teacher_model = BlackBoxTeacher(teacher_transformer , max_new_tokens=MAX_NEW_TOKENS)

    # Optimizer (nnx)
    optimizer = nnx.Optimizer(student_model , optax.adam(1e-3) , wrt=nnx.Param)

    # RNG
    rng , rng_data = jax.random.split(rng)
    keys = jax.random.split(rng_data , NUM_BATCHES)
    prompt_batches = [
        jax.random.randint(key , (BATCH , PROMPT_LEN) , 0 , config.VOCAB_SIZE , dtype=jnp.int32) for key in keys
    ]
    assert student_model(prompt_batches[0]).shape == (BATCH , PROMPT_LEN , config.VOCAB_SIZE) , f'Wrong: {student_model(prompt_batches[0]).shape}'

    # Full training
    rng , rng_train = jax.random.split(rng)
    train(
        teacher_model , proxy_model , student_model , optimizer , prompt_batches , rng_train , max_new_tokens=MAX_NEW_TOKENS , num_epochs=3
    )
    print('Mock run complete: Phase II Student-KD')