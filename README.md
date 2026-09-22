# recirculation-from-scratch
Build Recirculation at toy scale in PyTorch, following Mozer et al. 2026 (arXiv:2608.17981). Feedforward transformers can only push state up in depth, so a disambiguated deep residual is invisible to later tokens' shallow layers. Implement a training-free leak from a deep source into a shallow destination, contrast it with looping, and measure lag.
