import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformer import CausalLanguageModel, TransformerConfig
from utils import get_token_log_probs , collection , BlackBoxTeacher

ALPHA = 100.0 #paper value

def student_nll_loss(student_model , input_ids , teacher_response):
    # Equation 7
    token_log_probs = get_token_log_probs(student_model , input_ids , teacher_response)
    return -jnp.mean(token_log_probs.sum(-1))

def student_kl_loss(proxy_model , student_model , input_ids , teacher_response , weight):
    # Eq 8/10: dense full-vocab KL(proxy || student) , teacher-forced along teacher response y
    full_seq = jnp.concatenate([input_ids , teacher_response] , axis=-1)
    log_probs_proxy = jax.nn.log_softmax(proxy_model(full_seq) , axis=-1)[: , :-1 , :]
    log_probs_student = jax.nn.log_softmax(student_model(full_seq) , axis=-1)[: , :-1 , :]

    kl_per_pos = jnp.sum(jnp.exp(log_probs_proxy) * (log_probs_proxy - log_probs_student) , axis=-1)  # [batch , p+r-1]

    # Keep only the response , then per-sequence sum
    response_len = teacher_response.shape[-1]
    kl_per_seq = kl_per_pos[: , -response_len:].sum(-1)  # [batch]

    return jnp.mean(weight * kl_per_seq)

def proxy_seq_log_likelihood(proxy_model , input_ids , teacher_response):
    return get_token_log_probs(proxy_model , input_ids , teacher_response).sum(-1)  # [batch]

def sample_weight(seq_log_lik , mu , gamma):
    # Eq 9: sigmoid((log pi_p(y|x) - mu) / gamma) , frozen proxy so no grad
    return jax.nn.sigmoid((seq_log_lik - mu) / (gamma + 1e-8))  # [batch]

@nnx.jit
def train_step(proxy_model , student_model , optimizer , input_ids , teacher_response , weight):
    def loss_fn(student_model):
        nll_loss = student_nll_loss(student_model , input_ids , teacher_response)
        weighted_kl_loss = student_kl_loss(proxy_model , student_model , input_ids , teacher_response , weight)
        return nll_loss + ALPHA * weighted_kl_loss

    # Updates & Autograd
    loss , grads = nnx.value_and_grad(loss_fn)(student_model)
    optimizer.update(student_model , grads)
    return loss

def build_dataset(teacher_model , proxy_model , prompt_batches , rng , max_new_tokens):
    # proxy is frozen in phase ii: one collection pass = a fixed Ds
    dataset = []
    log_liks = []
    for x in prompt_batches:
        rng , rng_collection = jax.random.split(rng)
        _ , teacher_response , _ = collection(teacher_model , proxy_model , x , rng_collection , max_new_tokens)
        seq_log_lik = proxy_seq_log_likelihood(proxy_model , x , teacher_response)
        dataset.append((x , teacher_response , seq_log_lik))
        log_liks.append(seq_log_lik)

    # Global mean / std over Ds for the Eq 9 weights
    all_log_liks = jnp.concatenate(log_liks)
    mu , gamma = jnp.mean(all_log_liks) , jnp.std(all_log_liks)
    return dataset , mu , gamma

def train_epoch(proxy_model , student_model , optimizer , dataset , mu , gamma):
    losses = []
    for x , teacher_response , seq_log_lik in dataset:
        weight = sample_weight(seq_log_lik , mu , gamma)
        loss = train_step(proxy_model , student_model , optimizer , x , teacher_response , weight)
        losses.append(float(loss))

    return sum(losses) / len(losses)

def train(teacher_model , proxy_model , student_model , optimizer , prompt_batches , rng , max_new_tokens , num_epochs=3):
    rng , rng_data = jax.random.split(rng)
    dataset , mu , gamma = build_dataset(teacher_model , proxy_model , prompt_batches , rng_data , max_new_tokens)

    for epoch in range(num_epochs):
        mean_loss = train_epoch(proxy_model , student_model , optimizer , dataset , mu , gamma)
        print(f'Epoch: {epoch+1} , avg_loss: {mean_loss:.4f}')

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