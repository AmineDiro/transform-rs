use anyhow::{Ok, Result};
use candle_core::Tensor;
use candle_nn::{
    embedding, layer_norm, linear, Dropout, Embedding, LayerNorm, LayerNormConfig, Linear,
    VarBuilder,
};

pub struct Config {
    max_seq_len: usize,
    vocab_size: usize,
    n_embd: usize,
    n_heads: usize,
    n_layers: usize,
    dropout: Option<f32>,
    train: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_seq_len: 256,
            vocab_size: 32768,
            n_embd: 384,
            n_heads: 6,
            n_layers: 6,
            dropout: Some(0.2),
            train: true,
        }
    }
}
struct Head {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dropout: Dropout,
}
struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: Dropout,
}

struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
}
struct Block {
    mh_attention: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}
impl Block {
    fn build() -> Result<Self> {
        todo!()
    }
    fn forward(&mut self, x: Tensor) -> Result<Tensor> {
        todo!()
    }
}

struct GPT {
    token_embedding: Embedding,
    positional_embedding: Embedding,
    blocks: Vec<Block>,
    layer_norm: LayerNorm,
    gpt_head: Linear,
}

impl GPT {
    fn build(config: &Config, vb: VarBuilder) -> Result<Self> {
        let token_embedding =
            embedding(config.vocab_size, config.n_embd, vb.pp("token_embedding"))?;
        let positional_embedding = embedding(
            config.vocab_size,
            config.n_embd,
            vb.pp("postional_embedding"),
        )?;
        let blocks = vec![];
        let ln_config = LayerNormConfig::default();
        let layer_norm = layer_norm(config.n_embd, ln_config, vb.pp("layer_norm"))?;
        let gpt_head = linear(config.n_embd, config.vocab_size, vb.pp("gpt_head"))?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            layer_norm,
            gpt_head,
        })
    }
    fn forward(&mut self, xs: Tensor) -> Result<Tensor> {
        let (bs, seq_len) = xs.dims2()?;
        let tok_emb = xs.apply(&self.token_embedding)?;
        let pos_emb = xs.apply(&self.positional_embedding)?;
        let mut x = (tok_emb + pos_emb)?; // (B, T, C)
        for block in &mut self.blocks {
            x = block.forward(x)?; // (B, T, C)
        }
        x = x.apply(&self.layer_norm)?; // (B, T, C)
        let logits = x.apply(&self.gpt_head)?; // (B, T, vocab_size)
        Ok(logits)
    }
}
