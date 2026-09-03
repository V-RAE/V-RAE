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
    from vrae.models.dit.rope import apply_rope, RoPE3D

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
                self.qkv.weight.copy_(reference.qkv.weight)
                self.qkv.bias.copy_(reference.qkv.bias)
                self.proj.load_state_dict(reference.proj.state_dict())
                self.q_norm.load_state_dict(reference.q_norm.state_dict())
                self.k_norm.load_state_dict(reference.k_norm.state_dict())

        def forward(self, value, freqs, attention_mask=None):
            batch, length, _ = value.shape
            qkv = self.qkv(value).reshape(batch, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = self.q_norm(q.transpose(1, 2))
            k = self.k_norm(k.transpose(1, 2))
            v = v.transpose(1, 2)
            q = apply_rope(q.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
            k = apply_rope(k.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
            q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            return self.proj(result.transpose(1, 2).reshape(batch, length, self.dim))

    class FusedMLP(SwiGLUFeedForward):
        def __init__(self, reference: SwiGLUFeedForward) -> None:
            super().__init__(reference.w12.in_features, reference.w12.out_features // 2)
            with torch.no_grad():
                self.w12.weight.copy_(reference.w12.weight)
                self.w12.bias.copy_(reference.w12.bias)
                self.w3.load_state_dict(reference.w3.state_dict())

        def forward(self, value):
            first, second = self.w12(value).chunk(2, dim=-1)
            return self.w3(torch.nn.functional.silu(first) * second)

    device = torch.device("cuda")
    torch.manual_seed(7)
    baseline = AdaLNZeroDDTBlock(1536, 24, 4.0, attention_backend="sdpa").to(device).eval()
    qkv_fused = copy.deepcopy(baseline)
    qkv_fused.attn = FusedAttention(baseline.attn).to(device).eval()
    mlp_fused = copy.deepcopy(baseline)
    mlp_fused.mlp = FusedMLP(baseline.mlp).to(device).eval()
    fused = copy.deepcopy(qkv_fused)
    fused.mlp = FusedMLP(baseline.mlp).to(device).eval()
    value = torch.randn(2, 1280, 1536, device=device, dtype=torch.bfloat16)
    condition = torch.randn(2, 1536, device=device, dtype=torch.bfloat16)
    freqs = RoPE3D(64, end=16)(5, 16, 16).to(device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline(value, condition, freqs)
        qkv_fused(value, condition, freqs)
        mlp_fused(value, condition, freqs)
        fused(value, condition, freqs)
    torch.cuda.synchronize()

    def run(block):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = block(value, condition, freqs)
        end.record()
        torch.cuda.synchronize()
        return output, start.elapsed_time(end)

    base_output, base_ms = run(baseline)
    qkv_output, qkv_ms = run(qkv_fused)
    mlp_output, mlp_ms = run(mlp_fused)
    fused_output, fused_ms = run(fused)
    for optimized_output in (qkv_output, mlp_output, fused_output):
        torch.testing.assert_close(optimized_output, base_output, rtol=0.0, atol=2.0e-2)
    max_error = (base_output.float() - fused_output.float()).abs().max().item()
    mean_error = (base_output.float() - fused_output.float()).abs().mean().item()
    train_value_base = value.detach().clone().requires_grad_(True)
    train_value_fused = value.detach().clone().requires_grad_(True)
    train_condition = condition.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline(train_value_base, train_condition, freqs).float().square().mean().backward()
        fused(train_value_fused, train_condition, freqs).float().square().mean().backward()
    input_grad_error = (train_value_base.grad - train_value_fused.grad).abs().max().item()
    torch.testing.assert_close(train_value_fused.grad, train_value_base.grad, rtol=0.0, atol=5.0e-2)
    print(
        f"baseline_cuda_ms={base_ms:.3f} qkv_fused_cuda_ms={qkv_ms:.3f} "
        f"mlp_fused_cuda_ms={mlp_ms:.3f} both_fused_cuda_ms={fused_ms:.3f}"
    )
    print(
        f"qkv_speedup={base_ms / qkv_ms:.3f} mlp_speedup={base_ms / mlp_ms:.3f} "
        f"both_speedup={base_ms / fused_ms:.3f}"
    )
    print(f"max_abs_error={max_error:.6g} mean_abs_error={mean_error:.6g} input_grad_max_error={input_grad_error:.6g}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sdpa_vs_flash_attention() -> None:
    """Compare SDPA and flash-attn v2 with identical block weights and inputs."""
    from flash_attn import flash_attn_func
    from vrae.models.dit.blocks import AdaLNZeroDDTBlock, QKNormAttention
    from vrae.models.dit.rope import apply_rope, RoPE3D

    class FlashAttention(QKNormAttention):
        def __init__(self, reference: QKNormAttention) -> None:
            super().__init__(
                reference.dim,
                reference.num_heads,
                rope_theta=reference.rope_theta,
                attention_dropout=reference.attention_dropout,
                attention_backend="sdpa",
            )
            self.load_state_dict(reference.state_dict())

        def forward(self, value, freqs, attention_mask=None):
            batch, length, _ = value.shape
            qkv = self.qkv(value).reshape(batch, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = apply_rope(q.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
            k = apply_rope(k.transpose(1, 2).reshape(batch, length, self.dim), freqs, self.num_heads)
            q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            if attention_mask is not None or (self.training and self.attention_dropout):
                raise AssertionError("flash-attn comparison requires no mask/dropout")
            # flash-attn v2 uses [B, L, H, Dh], while the block stores [B, H, L, Dh].
            result = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=0.0,
                causal=False,
            )
            result = result.transpose(1, 2).reshape(batch, length, self.dim)
            return self.proj(result)

    device = torch.device("cuda")
    torch.manual_seed(11)
    sdpa = AdaLNZeroDDTBlock(1536, 24, 4.0, attention_backend="sdpa").to(device).eval()
    flash = copy.deepcopy(sdpa)
    flash.attn = FlashAttention(sdpa.attn).to(device).eval()
    value = torch.randn(2, 1280, 1536, device=device, dtype=torch.bfloat16)
    condition = torch.randn(2, 1536, device=device, dtype=torch.bfloat16)
    freqs = RoPE3D(64, end=16)(5, 16, 16).to(device=device)

    def run(block, repetitions=20):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                block(value, condition, freqs)
        torch.cuda.synchronize()
        timings = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = block(value, condition, freqs)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        return output, sum(timings) / len(timings)

    sdpa_output, sdpa_ms = run(sdpa)
    flash_output, flash_ms = run(flash)
    max_error = (sdpa_output.float() - flash_output.float()).abs().max().item()
    mean_error = (sdpa_output.float() - flash_output.float()).abs().mean().item()
    torch.testing.assert_close(flash_output, sdpa_output, rtol=0.0, atol=3.0e-2)

    sdpa_value = value.detach().clone().requires_grad_(True)
    flash_value = value.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sdpa(sdpa_value, condition, freqs).float().square().mean().backward()
        flash(flash_value, condition, freqs).float().square().mean().backward()
    grad_error = (sdpa_value.grad - flash_value.grad).abs().max().item()
    torch.testing.assert_close(flash_value.grad, sdpa_value.grad, rtol=0.0, atol=8.0e-2)
    print(
        f"sdpa_cuda_ms={sdpa_ms:.3f} flash_attn_cuda_ms={flash_ms:.3f} "
        f"speedup={sdpa_ms / flash_ms:.3f}"
    )
    print(
        f"max_abs_error={max_error:.6g} mean_abs_error={mean_error:.6g} "
        f"input_grad_max_error={grad_error:.6g}"
    )
