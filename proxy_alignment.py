# dataset: x (input_ids)
import jax
import jax.numpy as jnp
from flax import nnx

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

def generate(model , input_ids , rng):
    # wnat to return the probability distribution and hard output
    # NOTE: must replace black box model with a true API (irl, we wont get a prob distribution form a black box model)

    logits = model(input_ids)  # [batch , prompt_len , vocab_size]
    log_probs = jax.nn.log_softmax(logits , axis=1)
    next_token = jax.random.categorical(rng , logits[: , -1 , :] , axis=-1)
    return next_token , log_probs


def preference_loss(proxy_model , proxy_model_old , input_ids , y_winner , y_loser):
    # from Per-Token to Per-Sequence by summing the sequence length
    log_probs_winner = get_token_log_probs(proxy_model , input_ids , y_winner).sum(-1)  #[b]
    log_probs_loser = get_token_log_probs(proxy_model , input_ids , y_loser).sum(-1)
    log_probs_winner_old = get_token_log_probs(proxy_model_old , input_ids , y_winner).sum(-1)
    log_probs_loser_old = get_token_log_probs(proxy_model_old , input_ids , y_loser).sum(-1)

    # Ratios
    log_ratio_winner = log_probs_winner - log_probs_winner_old
    log_ratio_loser = log_ratio_loser - log_probs_loser_old

    # Logits
    logits_dpo = BETA * (log_ratio_winner - log_probs_loser)
    return -jnp.mean(jax.nn.log_sigmoid(logits_dpo))  # scalar

def proxy_nll_loss(proxy_model , input_ids , teacher_response):
    token_log_probs = get_token_log_probs(proxy_model , input_ids , teacher_response)  # [batch , response_len]
    return -jnp.mean(token_log_probs.sum(-1))  # scalar

@jax.jit
def train_step(state , state_old , batch):
    x , y_winner , y_loser = batch

    # Merges
    proxy_model , optimizer = nnx.merge(graphdef , state)
    proxy_model_old = nnx.merge(graphdef_proxy , state_old)

    # Loss
    def loss_fn(proxy_model):
        dpo_loss = preference_loss(proxy_model , proxy_model_old , x , y_winner , y_loser)
        nll_loss = proxy_nll_loss(proxy_model , x , y_winner)
        return dpo_loss + nll_loss

    # Updates
    loss , grads = nnx.value_and_grad(loss_fn)(proxy_model)
    optimizer.update(proxy_model , grads)

    _ , new_state = nnx.split((proxy_model , optimizer))
    return new_state , loss




    

