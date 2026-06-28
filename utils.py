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