"""Numerical-equivalence and speed comparison for block optimizations.

Run with: ``PYTHONPATH=src pytest -s -q test/test_dit_block_optimizations.py``.
"""

from __future__ import annotations

import copy
import time

import pytest
import torch
from torch import nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_qkv_and_mlp_are_numerically_equivalent() -> None:
    from vrae.models.dit.blocks import AdaLNZeroDDTBlock, QKNormAttention, SwiGLUFeedForward

    class FusedAttention(QKNormAttention):
        def __init__(self, reference: QKNormAttention) -> None:
            super().__init__(
                reference.dim,
                reference.num_heads,
                rope_theta=reference.rope_theta,
                attention_dropout=reference.attention_dropout,
                attention_backend=reference.attention_backend,
            )
            self.qkv = nn.Linear(reference.dim, reference.dim * 3)
            with torch.no_grad():
                self.qkv.weight.copy_(torch.cat((reference.q.weight, reference.k.weight, reference.v.weight), dim=0))
                self.qkv.bias.copy_(torch.cat((reference.q.bias, reference.k.bias, reference.v.bias), dim=0))
                self.proj.load_state_dict(reference.proj.state_dict())
                self.q_norm.load_state_dict(reference.q_norm.state_dict())
                self.k_norm.load_state_dict(reference.k_norm.state_dict())

        def forward(self, value, positions, attention_mask=None, *, rope_cache=None):
            batch, length, _ = value.shape
            qkv = self.qkv(value).reshape(batch, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = self.q_norm(q.transpose(1, 2))
            k = self.k_norm(k.transpose(1, 2))
            v = v.transpose(1, 2)
            if rope_cache is None:
                from vrae.models.rope3d import apply_video_dit_3d_rope_to_positions
                q = apply_video_dit_3d_rope_to_positions(q, positions, theta=self.rope_theta)
                k = apply_video_dit_3d_rope_to_positions(k, positions, theta=self.rope_theta)
            else:
                from vrae.models.rope3d import apply_video_dit_3d_rope_cache
                cosine, sine = rope_cache
                q = apply_video_dit_3d_rope_cache(q, cosine, sine)
                k = apply_video_dit_3d_rope_cache(k, cosine, sine)
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            return self.proj(result.transpose(1, 2).reshape(batch, length, self.dim))

    class FusedMLP(SwiGLUFeedForward):
        def __init__(self, reference: SwiGLUFeedForward) -> None:
            super().__init__(reference.w1.in_features, reference.w1.out_features)
            self.w12 = nn.Linear(reference.w1.in_features, reference.w1.out_features * 2)
            with torch.no_grad():
                self.w12.weight.copy_(torch.cat((reference.w1.weight, reference.w2.weight), dim=0))
                self.w12.bias.copy_(torch.cat((reference.w1.bias, reference.w2.bias), dim=0))
                self.w3.load_state_dict(reference.w3.state_dict())

        def forward(self, value):
            first, second = self.w12(value).chunk(2, dim=-1)
            return self.w3(torch.nn.functional.silu(first) * second)

    device = torch.device("cuda")
    torch.manual_seed(7)
    baseline = AdaLNZeroDDTBlock(1536, 24, 4.0, attention_backend="sdpa").to(device).eval()
    fused = copy.deepcopy(baseline)
    fused.attn = FusedAttention(baseline.attn).to(device).eval()
    fused.mlp = FusedMLP(baseline.mlp).to(device).eval()
    value = torch.randn(2, 1280, 1536, device=device, dtype=torch.bfloat16)
    condition = torch.randn(2, 1536, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(1280, 3, device=device, dtype=torch.long)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline(value, condition, positions)
        fused(value, condition, positions)
    torch.cuda.synchronize()

    def run(block):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = block(value, condition, positions)
        end.record()
        torch.cuda.synchronize()
        return output, start.elapsed_time(end)

    base_output, base_ms = run(baseline)
    fused_output, fused_ms = run(fused)
    max_error = (base_output.float() - fused_output.float()).abs().max().item()
    mean_error = (base_output.float() - fused_output.float()).abs().mean().item()
    torch.testing.assert_close(fused_output, base_output, rtol=0.0, atol=2.0e-2)
    train_value_base = value.detach().clone().requires_grad_(True)
    train_value_fused = value.detach().clone().requires_grad_(True)
    train_condition = condition.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline(train_value_base, train_condition, positions).float().square().mean().backward()
        fused(train_value_fused, train_condition, positions).float().square().mean().backward()
    input_grad_error = (train_value_base.grad - train_value_fused.grad).abs().max().item()
    torch.testing.assert_close(train_value_fused.grad, train_value_base.grad, rtol=0.0, atol=5.0e-2)
    print(f"baseline_cuda_ms={base_ms:.3f} fused_cuda_ms={fused_ms:.3f} speedup={base_ms / fused_ms:.3f}")
    print(f"max_abs_error={max_error:.6g} mean_abs_error={mean_error:.6g} input_grad_max_error={input_grad_error:.6g}")
