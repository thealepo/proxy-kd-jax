import jax
import jax.numpy as jnp

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

def collection(teacher_model , proxy_model , input_ids , rng , max_new_tokens):
    rng , rng_teacher , rng_proxy = jax.random.split(rng , 3)

    # Responses
    teacher_response = teacher_model.generate(input_ids , rng_teacher)  # real API call IRL
    proxy_full = autoregressive_generation(proxy_model , input_ids , rng_proxy , max_new_tokens=max_new_tokens)  # [batch , prompt_len+max_new_tokens]
    proxy_response = proxy_full[: , input_ids.shape[1]:]  # [batch , max_new_tokens]

    return input_ids , teacher_response , proxy_response  # x , y_winner , y_loser

# NOTE: FAKE CLASSSSSSSSSSSSS
class BlackBoxTeacher:
    def __init__(self , model , max_new_tokens):
        self.model = model
        self.max_new_tokens = max_new_tokens
    def generate(self , input_ids , rng):
        full_generation = autoregressive_generation(self.model , input_ids , rng , max_new_tokens=self.max_new_tokens)
        return full_generation[: , input_ids.shape[1]:]