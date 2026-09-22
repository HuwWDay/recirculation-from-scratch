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
- [ ] **9.** convex_mix
- [ ] **10.** nonconvex_mix
- [ ] **11.** no_normalization_mix
- [ ] **12.** recirculate_one_position
- [ ] **13.** ramped_alpha
- [ ] **14.** sequential_prefill
- [ ] **15.** insert_loop
- [ ] **16.** run_looped
- [ ] **17.** tied_lm_head
- [ ] **18.** ntp_loss
- [ ] **19.** perplexity
- [ ] **20.** concat_residuals
- [ ] **21.** scalar_mix_mlp
- [ ] **22.** vector_mix_mlp
- [ ] **23.** hadamard_mix
- [ ] **24.** adaptive_recirculate
- [ ] **25.** blockwise_recirculate
- [ ] **26.** lag_diagnostic
- [ ] **27.** frozen_stack_adaptive_demo

---

Built on Deep-ML.
