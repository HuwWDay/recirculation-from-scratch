# Recirculation from Scratch

Build Recirculation at toy scale in PyTorch, following Mozer et al. 2026 (arXiv:2608.17981). Feedforward transformers can only push state up in depth, so a disambiguated deep residual is invisible to later tokens' shallow layers. Implement a training-free leak from a deep source into a shallow destination, contrast it with looping, and measure lag.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rms_norm
- [x] **2.** causal_self_attention
- [x] **3.** gelu_ffn
- [x] **4.** pre_norm_block
- [x] **5.** embed_tokens
- [x] **6.** run_layers
- [x] **7.** last_axis_l2
- [x] **8.** match_source_norm
- [x] **9.** convex_mix
- [x] **10.** nonconvex_mix
- [x] **11.** no_normalization_mix
- [x] **12.** recirculate_one_position
- [x] **13.** ramped_alpha
- [x] **14.** sequential_prefill
- [x] **15.** insert_loop
- [x] **16.** run_looped
- [x] **17.** tied_lm_head
- [x] **18.** ntp_loss
- [x] **19.** perplexity
- [x] **20.** concat_residuals
- [x] **21.** scalar_mix_mlp
- [x] **22.** vector_mix_mlp
- [x] **23.** hadamard_mix
- [x] **24.** adaptive_recirculate
- [x] **25.** blockwise_recirculate
- [ ] **26.** lag_diagnostic
- [ ] **27.** frozen_stack_adaptive_demo

---

Built on Deep-ML.
