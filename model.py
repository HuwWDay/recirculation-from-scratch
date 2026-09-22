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

# Step 20 - concat_residuals
def concat_residuals(s, d):
    """Concatenate source and destination residuals along the last axis."""
    # TODO: Concatenate source and destination residuals along the last axis...
    return torch.cat([s, d], dim=-1)

# Step 21 - scalar_mix_mlp
def scalar_mix_mlp(concat_sd, mixer):
    """Produce scalar mixture coefficients (alpha, beta) from a concatenated residual."""
    # TODO: Produce scalar mixture coefficients (alpha, beta) from a concatenated residual.
    out = torch.nn.functional.layer_norm(concat_sd, (concat_sd.shape[-1],), weight=mixer['ln_weight'], bias=mixer['ln_bias'], eps=1e-5)
    out = torch.nn.functional.gelu(out @ mixer['w1'] + mixer['b1'])
    out = torch.nn.functional.gelu(out @ mixer['w2'] + mixer['b2'])
    out = torch.sigmoid(out @ mixer["w_out"] + mixer["b_out"])
    return out[..., 0:1], out[..., 1:2]

# Step 22 - vector_mix_mlp
def vector_mix_mlp(concat_sd, mixer):
    """Map concat(s, d) through a LayerNorm-GELU MLP to vector mixture coefficients of length D."""
    # TODO: Implement vector_mix_mlp to produce a pair of vector-valued mixture coefficients...
    x = concat_sd
    x = torch.nn.functional.layer_norm(x, (x.shape[-1],), mixer['ln_weight'], mixer['ln_bias'], 1e-5)
    x = x @ mixer['w1'] + mixer['b1'] 
    x = torch.nn.functional.gelu(x)
    x = x @ mixer['w2'] + mixer['b2'] 
    x = torch.nn.functional.gelu(x)
    x = x @ mixer["w_out"] + mixer["b_out"]
    x = torch.sigmoid(x)
    D = x.shape[-1] // 2
    return x[..., :D], x[..., D:]

# Step 23 - hadamard_mix
def hadamard_mix(s, d, alpha, beta):
    """Hadamard mix of matched source and destination."""
    f = match_source_norm(s, d)
    return alpha*f + beta*d

# Step 24 - adaptive_recirculate
def adaptive_recirculate(s, d, mixer):
    """Token-conditional vector mix of matched source into destination."""
    conc = concat_residuals(s, d)
    alpha, beta = vector_mix_mlp(conc, mixer)
    return hadamard_mix(s, d, alpha, beta)

# Step 25 - blockwise_recirculate
def blockwise_recirculate(embeddings, blocks, source_layer, dest_layer, alpha, block_size):
    """First-pass then mix K positions at a time and continue from dest."""
    B, T, D = embeddings.shape
    
    # 1. First-pass over the entire sequence
    residuals = run_layers(embeddings, blocks)
    # Ensure list representation and avoid in-place mutation of embeddings
    residuals = [r.clone() for r in residuals]

    # 2. Iterate in contiguous chunks of size block_size
    for start in range(0, T, block_size):
        end = min(start + block_size, T)

        # Extract the chunk at source and destination layers
        s = residuals[source_layer][:, start:end]
        d = residuals[dest_layer][:, start:end]

        # Match source norm to dest norm, then mix
        s_matched = match_source_norm(s, d)
        mixed = convex_mix(s_matched, d, alpha)

        # Update the destination layer slice
        residuals[dest_layer][:, start:end] = mixed

        # Re-run all downstream blocks starting from dest_layer
        curr = residuals[dest_layer]
        for i, block in enumerate(blocks[dest_layer:], start=dest_layer + 1):
            curr = pre_norm_block(curr, block)
            residuals[i] = curr

    return residuals

# Step 26 - lag_diagnostic
import torch
import torch.nn.functional as F

def lag_diagnostic(embeddings, tokens, blocks, embedding_weight, t, k, source_layer, dest_layer, alpha):
    """Change in next-token log-likelihood at lag k after recirculating position t."""
    # 1. Baseline forward pass with no recirculation
    baseline_residuals = run_layers(embeddings, blocks)
    # Ensure clone to prevent in-place aliasing
    baseline_residuals = [r.clone() for r in baseline_residuals]

    # 2. Recirculate at position t
    recirc_residuals = recirculate_one_position(
        baseline_residuals,
        t,
        source_layer,
        dest_layer,
        alpha,
        blocks
    )

    # 3. Position evaluated and target token index
    eval_pos = t + k
    target_tokens = tokens[:, eval_pos + 1].unsqueeze(-1)  # shape: (B, 1)

    # 4. Final residual states at position t+k: (B, 1, D)
    h_baseline = baseline_residuals[-1][:, eval_pos:eval_pos + 1]
    h_recirc = recirc_residuals[-1][:, eval_pos:eval_pos + 1]

    # 5. Project through tied LM head to obtain logits: (B, 1, V)
    logits_baseline = tied_lm_head(h_baseline, embedding_weight)
    logits_recirc = tied_lm_head(h_recirc, embedding_weight)

    # 6. Compute log-probabilities across vocabulary: (B, 1, V)
    log_probs_baseline = F.log_softmax(logits_baseline, dim=-1)
    log_probs_recirc = F.log_softmax(logits_recirc, dim=-1)

    # 7. Gather log-likelihood of target tokens: (B, 1)
    nll_baseline = log_probs_baseline.squeeze(1).gather(dim=-1, index=target_tokens)
    nll_recirc = log_probs_recirc.squeeze(1).gather(dim=-1, index=target_tokens)

    # 8. Mean change in log-likelihood (recirc - baseline)
    delta_ll = (nll_recirc - nll_baseline).mean()

    return delta_ll

# Step 27 - frozen_stack_adaptive_demo (not yet solved)
# TODO: implement

