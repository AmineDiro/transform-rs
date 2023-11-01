# Candle port of ng-lecture

This is a learning project.
I wanted to implement the transformer architecture from scratch and progressively add the recent tricks like : FlashAttention, KV Cache, sliding window attention, RoPE, Grouped Query Attention...
I also wanted to test `candle` library and see how ergonomic it is to write NN in Rust.

## Todo:

- [ ] Implement GPT architecture from [ng-lecture](https://github.com/karpathy/ng-video-lecture)
  - [ ] Define all layers
  - [ ] Implement attention mechanism
  - [ ] Optimize attention mechanism
- [ ] Train on Tinystories dataset
- [ ] Implement KV cache
- [ ] Implement Flash Attention
- [ ] Extend window
- [ ] Implement quantization
