import jax
import jax.numpy as jnp
from flax import nnx
import optax

from transformer import TransformerConfig , CausalLanguageModel
from utils import BlackBoxTeacher , autoregressive_generation
import proxy_alignment as phase1
import student_kd as phase2

# globabl prompt creator
def make_prompts(rng , num_batches , batch , prompt_len , vocab_size):
    keys = jax.random.split(rng , num_batches)
    return [
        jax.random.randint(key , (batch,prompt_len) , 0 , vocab_size , dtype=jnp.int32) for key in keys
    ]