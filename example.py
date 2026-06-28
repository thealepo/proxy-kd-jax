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

    # Main RNG
    rng = jax.random.PRNGKey(42)

    # Building the three models
    teacher_transformer = CausalLanguageModel(teacher_config , rngs=nnx.Rngs(0))
    teacher_model = BlackBoxTeacher(teacher_transformer , max_new_tokens=MAX_NEW_TOKENS)
    proxy_model = CausalLanguageModel(proxy_config , rngs=nnx.Rngs(1))
    student_model = CausalLanguageModel(student_config , rngs=nnx.Rngs(2))

    # Shared prompt set
    rng , rng_data = jax.random.split(rng)
    prompt_batches = make_prompts(rng_data , NUM_BATCHES , BATCH , PROMPT_LEN , VOCAB_SIZE)


    # PHASE I - Align the Proxy to the Black-Box Teacher (DPO + NLL)
    print('=== PHASE I: PROXY ALIGNMENT ===')
    proxy_optimizer = nnx.Optimizer(proxy_model , optax.adam(1e-3) , wrt=nnx.Param)
    rng , rng_phase1 = jax.random.split(rng)
    proxy_model = phase1.train(
        teacher_model , proxy_model , proxy_optimizer , prompt_batches , rng_phase1 ,
        max_new_tokens=MAX_NEW_TOKENS , num_epochs=10
    )

    # PHASE II - Distill the Aligned Proxy into the Student (NLL + Weighted KL)
    print('\n=== PHASE II: Student Distillation ===')
    student_optimizer = nnx.Optimizer(student_model , optax.adam(1e-3) , wrt=nnx.Param)
    rng , rng_phase2 = jax.random.split(rng)
    student_model = phase2.train(
        teacher_model , proxy_model , student_model , student_optimizer , prompt_batches , rng_phase2 ,
        max_new_tokens=MAX_NEW_TOKENS , num_epochs=10
    )