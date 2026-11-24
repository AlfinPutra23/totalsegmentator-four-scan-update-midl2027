"""Minimal LoRA for 3-D convolutions (vccor/peft, pre-submission block B3). Pure torch; no file access.

LoRAConv3d wraps an nn.Conv3d `base` (frozen) and adds a rank-r update of its weight viewed as a matrix
W in R^{C_out x (C_in/groups * k1 * k2 * k3)}:
        W' = W + (alpha / r) * B @ A,   A in R^{r x C_in*k^3} (Kaiming-uniform init, a = sqrt(5)),  B in R^{C_out x r} (zeros)
so the wrapped layer equals the base layer at initialisation (B = 0) and only A, B train. forward() is
conv3d(x, W', bias, stride, padding, dilation, groups) - the base's own hyper-parameters; bias stays frozen.
inject(net, r, alpha) wraps EVERY nn.Conv3d of the network once (aliases share the adapter; encoder, decoder, the 1x1x1 heads); transposed
convolutions and normalisation layers stay frozen. merge(net) folds W' back into plain nn.Conv3d modules, so the merged
network has EXACTLY the original module tree / state_dict keys (TotalSegmentator's predictor loads it unchanged).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv3d(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        assert isinstance(base, nn.Conv3d) and r >= 1
        self.base, self.r, self.scaling = base, int(r), float(alpha) / float(r)
        for p in self.base.parameters():
            p.requires_grad_(False)
        fan_in = base.in_channels // base.groups * int(torch.tensor(base.kernel_size).prod())
        self.lora_A = nn.Parameter(torch.empty(self.r, fan_in, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, self.r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self):
        return (self.lora_B @ self.lora_A).view(self.base.weight.shape) * self.scaling

    def forward(self, x):
        b = self.base
        return F.conv3d(x, b.weight + self.delta(), b.bias, b.stride, b.padding, b.dilation, b.groups)

    def merged(self):
        b = self.base
        m = nn.Conv3d(b.in_channels, b.out_channels, b.kernel_size, b.stride, b.padding, b.dilation, b.groups,
                      bias=b.bias is not None, padding_mode=b.padding_mode, device=b.weight.device, dtype=b.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(b.weight + self.delta())
            if b.bias is not None:
                m.bias.copy_(b.bias)
        return m


def _replace(net, cls_from, fn, memo=None):
    """Replace every module of type cls_from by fn(module). MEMOISED by module identity: nnU-Net's ConvDropoutNormReLU
    registers the same conv twice (self.conv and self.all_modules[0]; forward runs through all_modules), so every
    reference to one module must receive the SAME replacement. Returns the memo {id(original): replacement}."""
    memo = {} if memo is None else memo
    for name, child in list(net.named_children()):
        if isinstance(child, cls_from):
            if id(child) not in memo:
                memo[id(child)] = fn(child)
            setattr(net, name, memo[id(child)])
        elif not isinstance(child, LoRAConv3d):
            _replace(child, cls_from, fn, memo)
    return memo


def inject(net, r, alpha):
    """Freeze every parameter, then wrap every (unique) nn.Conv3d with a LoRA adapter. Returns the number of adapters."""
    for p in net.parameters():
        p.requires_grad_(False)
    return len(_replace(net, nn.Conv3d, lambda c: LoRAConv3d(c, r, alpha)))


def lora_parameters(net):
    return [p for m in net.modules() if isinstance(m, LoRAConv3d) for p in (m.lora_A, m.lora_B)]


def merge(net):
    """Fold every adapter into a plain nn.Conv3d (in place, aliases kept shared); returns the number merged."""
    return len(_replace(net, LoRAConv3d, lambda m: m.merged()))


def count(net):
    lp = lora_parameters(net)
    return {"lora_params": int(sum(p.numel() for p in lp)), "adapters": len(lp) // 2,
            "trainable": int(sum(p.numel() for p in net.parameters() if p.requires_grad))}
