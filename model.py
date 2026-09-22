"""
Recirculation from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rms_norm
def rms_norm(x, gain, eps=1e-6):
    """Apply RMSNorm over the last dimension with a learnable gain vector and eps 1e-6."""
    # TODO: Implement rms_norm so that it normalizes x over the last dimension and then scales by gain.
    avg = torch.mean(x**2, dim=-1, keepdim=True)
    rms = torch.sqrt(avg + eps)
    return (x / rms)*gain

# Step 2 - causal_self_attention
def causal_self_attention(x, w_q, w_k, w_v, w_o):
    """Compute single-head causal scaled-dot-product attention."""
    # TODO: Compute single-head causal scaled-dot-product attention...
    B, T, D = x.shape
    Q, K, V = x @ w_q, x @ w_k, x @ w_v 
    scores = (Q @ K.transpose(-2, -1)) / D**0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return (attn @ V) @ w_o

# Step 3 - gelu_ffn
def gelu_ffn(x, w_ff1, w_ff2):
    """Apply a two-layer position-wise GELU feed-forward that expands to 4D then projects back to D."""
    # TODO: Apply a two-layer position-wise GELU FFN that expands to 4D then back to D...
    out = x @ w_ff1 
    out = torch.nn.functional.gelu(out)
    return out @ w_ff2

# Step 4 - pre_norm_block
def pre_norm_block(x, block):
    """Wrap attention and feed-forward as a pre-norm residual transformer block."""
    # 1. Pre-norm self-attention sub-layer with residual connection
    norm_attn = rms_norm(x, block["attn_gain"])
    attn_out = causal_self_attention(
        norm_attn, 
        block["w_q"], 
        block["w_k"], 
        block["w_v"], 
        block["w_o"]
    )
    x = x + attn_out

    # 2. Pre-norm feed-forward sub-layer with residual connection
    norm_ffn = rms_norm(x, block["ffn_gain"])
    ffn_out = gelu_ffn(norm_ffn, block["w_ff1"], block["w_ff2"])
    x = x + ffn_out

    return x

# Step 5 - embed_tokens
def embed_tokens(tokens, embedding_weight):
    """Embed token ids with a (V, D) table."""
    # TODO: Implement embed_tokens to produce a residual-stream vector for every token id.
    return torch.nn.functional.embedding(tokens, embedding_weight)

# Step 6 - run_layers
import torch

def run_layers(x, blocks):
    """Return residual streams after every layer as a stacked tensor."""
    out = [x]
    for block in blocks:
        out.append(pre_norm_block(out[-1], block))
    return torch.stack(out, dim=0)

# Step 7 - last_axis_l2
def last_axis_l2(x):
    """Return last-axis L2 norms of x with a kept singleton dimension."""
    # TODO: Compute the L2 norm of a tensor over its last axis keeping that axis as a singleton.
    return torch.linalg.norm(x, dim=-1, keepdim=True)

# Step 8 - match_source_norm
def match_source_norm(s, d):
    """Rescale s so its last-axis L2 matches d."""
    # Ensure keepdim=True so dimensions broadcast cleanly with s: (..., 1)
    norm_s = last_axis_l2(s)
    norm_d = last_axis_l2(d)

    # Elementwise condition: where s norm is 0, scale factor is 0
    scale = torch.where(norm_s == 0, torch.zeros_like(norm_s), norm_d / norm_s)
    return s * scale

# Step 9 - convex_mix
def convex_mix(s, d, alpha):
    """Convex mix of destination with a magnitude-matched source."""
    s = match_source_norm(s, d)
    return (1-alpha)*d+alpha*s

# Step 10 - nonconvex_mix
def nonconvex_mix(s, d, alpha):
    """Nonconvex mix: destination plus a scaled matched source."""
    s = match_source_norm(s, d)
    return d+alpha*s

# Step 11 - no_normalization_mix
def no_normalization_mix(s, d, alpha):
    """Mix source into destination with no renormalization using the raw source."""
    # TODO: Mix source into destination with no renormalization using the raw source.
    return (1-alpha)*d + alpha*s

# Step 12 - recirculate_one_position
def recirculate_one_position(residuals, t, source_layer, dest_layer, alpha, blocks):
    """Mix source into dest at time t then re-run blocks from dest onward."""
    s = residuals[source_layer][:, t]
    d = residuals[dest_layer][:, t]
    mix = convex_mix(s, d, alpha)

    # Clone the target residual and inject the mixed representation at position t
    x = residuals[dest_layer].clone()
    x[:, t] = mix

    # Start with a copy of all original residuals so shape and untruncated layers are preserved
    new_residuals = list(residuals)
    new_residuals[dest_layer] = x

    # Re-run the subsequent blocks, updating new_residuals from dest_layer + 1 onward
    curr = x
    for i, block in enumerate(blocks[dest_layer:], start=dest_layer + 1):
        curr = pre_norm_block(curr, block)
        if i < len(new_residuals):
            new_residuals[i] = curr
        else:
            new_residuals.append(curr)

    return new_residuals

# Step 13 - ramped_alpha
def ramped_alpha(t, alpha, ramp_steps=10):
    """Compute the ramped mixture coefficient for a 0-indexed token position t."""
    # TODO: Compute the ramped mixture coefficient for a 0-indexed token position t...
    return min(t/ramp_steps, 1)*alpha

# Step 14 - sequential_prefill
def sequential_prefill(embeddings, blocks, source_layer, dest_layer, alpha, ramp_steps=10):
    """Token-by-token recirculation prefill with ramped alpha."""
    B, T, D = embeddings.shape
    residuals = None

    for t in range(T):
        # 1. First-pass forward, cloning so in-place slice writes don't mutate input embeddings
        prefix_residuals = [r.clone() for r in run_layers(embeddings[:, :t + 1], blocks)]

        # 2. Write previously recirculated residuals back onto positions 0 .. t-1
        if t > 0:
            for l in range(len(prefix_residuals)):
                prefix_residuals[l][:, :t] = residuals[l][:, :t]

        # 3. Compute the ramped mixing parameter for step t
        current_alpha = ramped_alpha(t, alpha, ramp_steps)

        # 4. Recirculate at time t and re-run forward from dest_layer onward
        residuals = recirculate_one_position(
            prefix_residuals,
            t,
            source_layer,
            dest_layer,
            current_alpha,
            blocks
        )

    return residuals

# Step 15 - insert_loop
def insert_loop(blocks, l1, l2):
    """Insert a looped copy of blocks from l1+1 through l2 immediately after block l2."""
    # TODO: Build a new block list with one extra pass over layers l1+1 through l2.
    seg = blocks[l1+1:l2+1]
    pref = blocks[:l2+1]
    suff = blocks[l2+1:]
    return pref+seg+suff

# Step 16 - run_looped
def run_looped(x, blocks, l1, l2):
    """Run a looped stack and return the final residual."""
    newblock = insert_loop(blocks, l1, l2)
    return run_layers(x, newblock)[-1]

# Step 17 - tied_lm_head
def tied_lm_head(h, embedding_weight):
    """Project a residual stream to vocabulary logits with a tied embedding table."""
    # TODO: Project a residual stream to vocabulary logits with a tied embedding...
    embed_weight_t = embedding_weight.transpose(0, 1)
    return h @ embed_weight_t

# Step 18 - ntp_loss
def ntp_loss(logits, tokens):
    # TODO: Compute mean next-token-prediction cross-entropy of shifted (B, T, V) logits...
    B, T, V = logits.shape
    inp = logits[:, :-1, :].reshape(-1, V)
    target = tokens[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(inp, target)

# Step 19 - perplexity
def perplexity(loss):
    """Return exp(loss) for a scalar or tensor NTP cross-entropy."""
    # TODO: Compute the exponential of a next-token-prediction cross-entropy loss...
    if torch.is_tensor(loss):
        return torch.exp(loss)
    else:
        return math.exp(loss)

# Step 20 - concat_residuals (not yet solved)
# TODO: implement

# Step 21 - scalar_mix_mlp (not yet solved)
# TODO: implement

# Step 22 - vector_mix_mlp (not yet solved)
# TODO: implement

# Step 23 - hadamard_mix (not yet solved)
# TODO: implement

# Step 24 - adaptive_recirculate (not yet solved)
# TODO: implement

# Step 25 - blockwise_recirculate (not yet solved)
# TODO: implement

# Step 26 - lag_diagnostic (not yet solved)
# TODO: implement

# Step 27 - frozen_stack_adaptive_demo (not yet solved)
# TODO: implement

