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

# EXAMPLE OF USAGE
if __name__ == "__main__":
    
    # Shared token space: VOCAB_SIZE + SEQ_LEN must match across all 3 roles
    VOCAB_SIZE , SEQ_LEN = 256 , 32
    NUM_BATCHES , BATCH , PROMPT_LEN , MAX_NEW_TOKENS = 4 , 4 , 8 , 8

    # Three different configs: teacher is largest (BlackBox), student is the small model we want to distill into
    teacher_config = TransformerConfig(VOCAB_SIZE=VOCAB_SIZE , SEQ_LEN=SEQ_LEN , HIDDEN_SIZE=64 , N_HEADS=4 , N_LAYERS=2)
    proxy_config = TransformerConfig(VOCAB_SIZE=VOCAB_SIZE , SEQ_LEN=SEQ_LEN , HIDDEN_SIZE=32 , N_HEADS=4 , N_LAYERS=2)
    student_config = TransformerConfig(VOCAB_SIZE=VOCAB_SIZE , SEQ_LEN=SEQ_LEN , HIDDEN_SIZE=16 , N_HEADS=4 , N_LAYERS=1)