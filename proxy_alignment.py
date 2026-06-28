# dataset: x (input_ids)
import jax
import jax.numpy as jnp
from flax import nnx

BETA = 0.1

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


def preference_loss(state_proxy_model , state_proxy_model_old , input_ids , y_winner , y_loser):
    def get_token_log_probs(model , y):
        full_seq = jnp.concatenate([input_ids,y] , axis=-1)
        logits = model(full_seq)
        log_probs = jax.nn.log_softmax(logits , axis=-1)

        targets = full_seq[: , 1:]
        log_probs = log_probs[: , :-1 , :]

        token_log_probs = jnp.take_along_axis(
            log_probs , targets[... , jnp.newaxis] , axis=-1
        ).squeeze(-1)

        return token_log_probs

    # merging
    proxy = nnx.merge(graphdef_proxy , state_proxy_model)
    old_proxy = nnx.merge(graphdef_proxy , state_proxy_model_old)

    # log probs
    proxy_log_probs_winner = get_token_log_probs(proxy , y_winner)
    proxy_log_probs_loser = get_token_log_probs(proxy , y_loser)
    old_proxy_log_probs_winner = get_token_log_probs(old_proxy , y_winner)
    old_proxy_log_probs_loser = get_token_log_probs(old_proxy , y_loser)

    # DPO ratios
    log_ratio_winner = proxy_log_probs_winner - old_proxy_log_probs_winner
    log_ratio_loser = proxy_log_probs_loser - old_proxy_log_probs_loser

    # Loss
    logits_dpo = BETA * (log_ratio_winner - log_ratio_loser)

    return -jnp.mean(jax.nn.log_sigmoid(logits_dpo))

def proxy_nll_loss(teacher_label , proxy_distribution):
    label_proxy_prob = proxy_distribution(jnp.arange(teacher_label.shape[0]) , teacher_label)

    return -jnp.mean(jnp.log(label_proxy_prob))

def train_step(state , state_old , batch):
    x , y_winner , y_loser = batch

    # Reconstructing models
    proxy_model , optimizer = nnx.merge(graphdef , state)
    proxy_model_old = nnx.merge(graphdef , state_old)

    def loss_fn(proxy_model):
        dpo_loss = preference_loss(proxy_model , proxy_model_old , x , y_winner , y_loser)
        nll_loss = proxy_nll_loss(y_winner , y_loser_probs)
        return dpo_loss + nll_loss

    proxy_model = nnx.merge(graphdef_proxy , state_proxy_model)
    loss , grads = nnx.value_and_grad(loss_fn)(...)
    optimizer.update(proxy_model , grads)
    return state_proxy_model , loss

def collection(teacher_model , proxy_model , input_ids , rng):
    # rng splitting
    rng , rng_gen_teacher , rng_gen_proxy = jax.random.split(rng , )

    # sample responses
    teacher_response = teacher_model.generate(input_ids , rng_gen_teacher)  # NOTE: this prolly must be a true API
    proxy_response , proxy_distribution = generate(proxy_model , input_ids , rng_gen_proxy)


    

