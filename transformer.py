import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from dataclasses import dataclass

# Configuration for the Tranformer architecture to be used
@dataclass(frozen=True , kw_only=True , slots=True)
class TransformerConfig:
    VOCAB_SIZE = 256
    SEQ_LEN = 32
    HIDDEN_SIZE = 64
    MLP_HIDDEN_SIZE = 4 * 64
    N_HEADS = 4
    N_LAYERS = 2

# MHSA
class MultiHeadSelfAttention(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.n_heads = config.N_HEADS
        self.head_size = config.HIDDEN_SIZE // config.N_HEADS
        self.output_size = config.HIDDEN_SIZE
        self.hidden_size = config.HIDDEN_SIZE

        # Initializing the 4 Attention matrices (Query , Key, Value, Out)
        self.Wq = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wk = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wv = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)
        self.Wo = nnx.Linear(self.hidden_size , self.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        # x shape is [batch , seq_len , hidden_size]
        # QKV Values
        Q , K , V = self.Wq(x) , self.Wk(x) , self.Wv(x)  # [batch , seq_len , hidden_size]

        # To account for the multiple heads, we rearrange the shape of our tensors
        def mha_rearrange(t):
            return rearrange(t , 'b n (h d) -> b h n d' , h=self.n_heads)  # [batch , n_heads , seq_len , head_dim]
        Q , K , V = map(mha_rearrange , (Q,K,V))

        # Scale 1 / sqrt(head_dim)
        scale = (self.head_size) ** -0.5

        # Computing self-attention
        attention_weights = (jnp.einsum('b h i d , b h j d -> b h i j' , Q , K)) * scale  # QK^T / scale

        # Attention Causal map... to avoid tokens attending into future
        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len,seq_len) , dtype=jnp.bool_)) # creates a lower triangular matrix size seq_len*seq_len
        causal_mask = causal_mask[jnp.newaxis , jnp.newaxis , : , :]

        attention_weights = jnp.where(causal_mask , attention_weights , float('-inf'))
        attention_weights = jax.nn.softmax(attention_weights , axis=-1)
        out = jnp.einsum('b n i j , b n j d -> b n i d' , attention_weights , V)  # multiplying by V

        out = rearrange(out , 'b h n d -> b n (h d)') # Back to [batch , seq_len , hidden_size]
        out = self.Wo(out)

        return out

class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.HIDDEN_SIZE , config.MLP_HIDDEN_SIZE , rngs=rngs)
        self.fc2 = nnx.Linear(config.MLP_HIDDEN_SIZE , config.HIDDEN_SIZE , rngs=rngs)

    def __call__(self , x):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.mhsa = MultiHeadSelfAttention(config , rngs=rngs)
        self.mlp = MultiLayerPerceptron(config , rngs=rngs)
        self.ln1 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)

    def __call__(self , x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.wte = nnx.Embed(config.VOCAB_SIZE , config.HIDDEN_SIZE , rngs=rngs)
        self.wpe = nnx.Embed(config.SEQ_LEN , config.HIDDEN_SIZE , rngs=rngs)
        self.layers = nnx.List([TransformerLayer(config , rngs) for _ in range(config.N_LAYERS)])
        self.ln_f = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)  # final LayerNorm

    def __call__(self , input_ids):
        batch_size , seq_len = input_ids.shape  # [batch , seq_len]
        positions = jnp.arange(seq_len)

        # vanilla positonal embedding
        x = self.wte(input_ids) + self.wpe(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        return x  # [batch , seq_len , hidden_size]

# quick test
if __name__ == "__main__":
    config = TransformerConfig()
    model = Transformer(config , rngs=nnx.Rngs(0))

    input_ids = jnp.ones((4,32) , dtype=jnp.int32)
    x = model(input_ids)

    assert x.shape == (4,32,config.HIDDEN_SIZE) , f"Expected (4 , 32 , {config.HIDDEN_SIZE}) but got {x.shape}"
    print(f"Output shape: {x.shape}")