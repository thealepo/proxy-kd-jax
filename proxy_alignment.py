# dataset: x (input_ids)
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformer import Transformer , TransformerConfig , CausalLanguageModel

# NOTE: FAKE CLASSSSSSSSSSSSS
class BlackBoxTeacher:
    def __init__(self , model , max_new_tokens):
        self.model = model
        self.max_new_tokens = max_new_tokens
    def generate(self , input_ids , rng):
        full_generation = autoregressive_generation(self.model , input_ids , rng , max_new_tokens=self.max_new_tokens)
        return full_generation[: , input_ids.shape[1]:]

BETA = 0.1

def get_token_log_probs(model , input_ids , response):
    # input ids: [batch , prompt_len] | response: [batch , response_len]
    full_seq = jnp.concatenate([input_ids,response] , axis=-1)  # [batch , prompt+response]
    logits = model(full_seq)
    log_probs = jax.nn.log_softmax(logits , axis=-1)  # [batch , prompt+response , vocab_size]

    # Next-token pred
    targets = full_seq[: , 1:] # [batch , p+r-1]
    log_probs = log_probs[: , :-1 , :]  # [batch , p+r-1 , vocab_size]

    token_log_probs = jnp.take_along_axis(
        log_probs , targets[... , None] , axis=-1
    ).squeeze(-1)  # [batch , p+r-1]

    # Keep only the response
    response_len = response.shape[-1]
    return token_log_probs[: , -response_len:]  # [batch , response_len]

def autoregressive_generation(model , prompt , rng , max_new_tokens=256):
    batch , prompt_len = prompt.shape
    total_len = prompt_len + max_new_tokens

    # Buffer
    buffer = jnp.zeros((batch , total_len) , jnp.int32).at[: , :prompt_len].set(prompt)

    # body_fn for a fori_loop... decision to swtich
    def body_fn(i , carry):
        buffer , rng = carry
        rng , rng_gen = jax.random.split(rng)

        logits = model(buffer)  # [batch , total_len , vocab_size]
        next_token = jax.random.categorical(
            rng_gen , logits[: , prompt_len+i-1 , :] , axis=-1
        )  # [batch]
        buffer = buffer.at[: , prompt_len+i].set(next_token)
        return buffer , rng

    # Actual looping
    init_carry = (buffer , rng)
    buffer , _ = jax.lax.fori_loop(0 , max_new_tokens , body_fn , init_carry)
    return buffer

def preference_loss(proxy_model , proxy_model_old , input_ids , y_winner , y_loser):
    # from Per-Token to Per-Sequence by summing the sequence length
    log_probs_winner = get_token_log_probs(proxy_model , input_ids , y_winner).sum(-1)  #[b]
    log_probs_loser = get_token_log_probs(proxy_model , input_ids , y_loser).sum(-1)
    log_probs_winner_old = get_token_log_probs(proxy_model_old , input_ids , y_winner).sum(-1)
    log_probs_loser_old = get_token_log_probs(proxy_model_old , input_ids , y_loser).sum(-1)

    # Ratios
    log_ratio_winner = log_probs_winner - log_probs_winner_old
    log_ratio_loser = log_probs_loser - log_probs_loser_old

    # Logits
    logits_dpo = BETA * (log_ratio_winner - log_ratio_loser)
    return -jnp.mean(jax.nn.log_sigmoid(logits_dpo))  # scalar

def proxy_nll_loss(proxy_model , input_ids , teacher_response):
    token_log_probs = get_token_log_probs(proxy_model , input_ids , teacher_response)  # [batch , response_len]
    return -jnp.mean(token_log_probs.sum(-1))  # scalar

@nnx.jit
def train_step(proxy_model , optimizer , proxy_model_old , batch):
    x , y_winner , y_loser = batch

    # Loss fn
    def loss_fn(proxy_model):
        dpo_loss = preference_loss(proxy_model , proxy_model_old , x , y_winner , y_loser)
        nll_loss = proxy_nll_loss(proxy_model , x , y_winner)
        return dpo_loss + nll_loss

    # Updates and autograd
    loss , grads = nnx.value_and_grad(loss_fn)(proxy_model)
    optimizer.update(proxy_model , grads)
    return loss

def collection(teacher_model , proxy_model , input_ids , rng , max_new_tokens):
    rng , rng_teacher , rng_proxy = jax.random.split(rng , 3)

    # Responses
    teacher_response = teacher_model.generate(input_ids , rng_teacher)  # real API call IRL
    proxy_full = autoregressive_generation(proxy_model , input_ids , rng_proxy , max_new_tokens=max_new_tokens)  # [batch , prompt_len+max_new_tokens]
    proxy_response = proxy_full[: , input_ids.shape[1]:]  # [batch , max_new_tokens]

    return input_ids , teacher_response , proxy_response  # x , y_winner , y_loser

def snapshot(model):
    # creates a copy of a model at a specific iteration
    # this is for the previous iteration of the proxy model
    # as seen in equation 4 of the paper
    graphdef , state = nnx.split(model)
    return nnx.merge(graphdef , jax.tree.map(lambda a: a , state))

def train_epoch(teacher_model , proxy_model , optimizer , prompt_batches , rng , max_new_tokens):
    losses = []
    for x in prompt_batches:
        proxy_model_old = snapshot(proxy_model)  # previous-iteration reference (pi_old)

        # RNG dealing
        rng , rng_collection = jax.random.split(rng)

        # Run collection
        data = collection(
            teacher_model , proxy_model , x , rng_collection , max_new_tokens
        )

        # Run a train_step
        loss = train_step(proxy_model , optimizer , proxy_model_old , data)
        losses.append(float(loss))

    return sum(losses) / len(losses)

def train(teacher_model , proxy_model , optimizer , prompt_batches , rng , max_new_tokens , num_epochs=3):

    # Run # of epochs
    for epoch in range(num_epochs):
        rng , rng_epoch = jax.random.split(rng)
        mean_loss = train_epoch(
            teacher_model , proxy_model , optimizer , prompt_batches , rng_epoch , max_new_tokens
        )
        print(f'Epoch: {epoch+1} , avg_loss: {mean_loss:.4f}')

    return proxy_model


# MOCK RUN
if __name__ == "__main__":
    # Setup
    config = TransformerConfig()
    NUM_BATCHES , BATCH , PROMPT_LEN , MAX_NEW_TOKENS = 3 , 4 , 8 , 8

    # Setup v2 (RNGs, models, optimizer)
    rng = jax.random.PRNGKey(42)
    proxy_model = CausalLanguageModel(config , rngs=nnx.Rngs(0))
    teacher_transformer = CausalLanguageModel(config , rngs=nnx.Rngs(1))
    teacher_model = BlackBoxTeacher(teacher_transformer , max_new_tokens=MAX_NEW_TOKENS)
    optimizer = nnx.Optimizer(proxy_model , optax.adam(1e-3) , wrt=nnx.Param)

    # RTG
    rng , rng_data = jax.random.split(rng)
    keys = jax.random.split(rng_data , NUM_BATCHES)
    prompt_batches = [
        jax.random.randint(key , (BATCH , PROMPT_LEN) , 0 , config.VOCAB_SIZE , dtype=jnp.int32) for key in keys
    ]
    # TEST 1
    assert proxy_model(prompt_batches[0]).shape == (BATCH , PROMPT_LEN , config.VOCAB_SIZE) , f'wrong: {proxy_model(prompt_batches[0]).shape}'

    rng , rng_train = jax.random.split(rng)
    train(
        teacher_model , proxy_model , optimizer , prompt_batches , rng_train ,
        max_new_tokens=MAX_NEW_TOKENS , num_epochs=3
    )
    print('MOCK COMPLETE... PHASE I PROXY-KD')