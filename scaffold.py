"""
Recirculation from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Toy recirculation: sequential prefill vs looping vs a frozen-stack adaptive mixer."""
import torch


def _block(D):
    return {
        "attn_gain": torch.ones(D),
        "ffn_gain": torch.ones(D),
        "w_q": torch.zeros(D, D),
        "w_k": torch.zeros(D, D),
        "w_v": torch.zeros(D, D),
        "w_o": torch.zeros(D, D),
        "w_ff1": torch.zeros(D, 4 * D),
        "w_ff2": torch.zeros(4 * D, D),
    }


def main():
    torch.manual_seed(0)
    V, D, T = 8, 4, 6
    embedding_weight = torch.randn(V, D)
    blocks = [_block(D), _block(D)]
    tokens = torch.randint(0, V, (2, T))
    mixer = {
        "ln_weight": torch.ones(2 * D),
        "ln_bias": torch.zeros(2 * D),
        "w1": torch.zeros(2 * D, D),
        "b1": torch.zeros(D),
        "w2": torch.zeros(D, D),
        "b2": torch.zeros(D),
        "w_out": torch.zeros(D, 2 * D),
        "b_out": torch.zeros(2 * D),
    }
    base, fixed, adaptive = frozen_stack_adaptive_demo(
        tokens, embedding_weight, blocks, mixer,
        source_layer=2, dest_layer=1, alpha=0.15,
        steps=2, lr=0.05, seed=0,
    )
    print("baseline", float(base))
    print("fixed", float(fixed))
    print("adaptive", float(adaptive))
    e = embed_tokens(tokens, embedding_weight)
    looped = run_looped(e, blocks, 0, 1)
    print("looped", tuple(looped.shape))
    delta = lag_diagnostic(e, tokens, blocks, embedding_weight, 0, 1, 2, 1, 0.15)
    print("lag", float(delta))


if __name__ == "__main__":
    main()

