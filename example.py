import jax
import jax.numpy as jnp
from flax import nnx
import optax

from transformer import TransformerConfig , CausalLanguageModel
from utils import BlackBoxTeacher , autoregressive_generation
import proxy_alignment as phase1
import student_kd as phase2