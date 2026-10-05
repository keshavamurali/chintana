"""Mixture-of-Experts feed-forward layer, a drop-in replacement for chintana's MLP.

The dense chintana MLP is Linear(C, 4C) -> GELU -> Linear(4C, C). An MoE layer
holds n_experts copies of that shape (Expert, with the same c_fc / c_proj
parameter names, so GPT's scaled c_proj init and the dense -> MoE weight copy
both work by name) and a router Linear(C, n_experts) that picks top_k experts
per token:

  probs   = softmax(router(x))                    (N, E), fp32
  w, idx  = top_k(probs)                          (N, k)
  w       = w / w.sum(-1)        if normalize_topk
  y       = sum_j w_j * Expert_{idx_j}(x)

With normalize_topk the gates sum to 1, so when every expert holds the same
weights y == Expert(x) == dense MLP(x) exactly (up to float rounding). That is
what makes upcycling a function-preserving conversion (see model.upcycle).

Tokens are dispatched by gathering each expert's tokens, running the expert
on just those rows and scattering the weighted outputs back with index_add_,
so an expert's compute is proportional to the tokens it actually receives.
With capacity_factor > 0 an expert keeps at most
ceil(capacity_factor * N * k / E) assignments (in token order); the rest are
dropped and that token's slot contributes nothing (the residual stream still
carries the token past the layer).

After every forward the layer leaves its auxiliary losses and routing counts
in attributes, which MoEGPT collects:
  aux_loss  Switch-Transformer load-balancing loss  E * sum_i f_i * P_i
            (f_i = fraction of assignments sent to expert i, P_i = mean router
            prob of expert i); 1.0 when routing is perfectly uniform.
  z_loss    mean(logsumexp(router logits)^2) (ST-MoE), keeps logits small.
  counts    assignments per expert, (E,) long, before capacity dropping.
  dropped   assignments dropped by the capacity limit.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Expert(nn.Module):
    """chintana.mlp.MLP with a configurable hidden width."""

    def __init__(self, n_embd, hidden, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class MoE(nn.Module):

    def __init__(self, n_embd, n_experts, top_k, hidden, bias=True, dropout=0.0,
                 normalize_topk=True, capacity_factor=0.0, router_jitter=0.0):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.capacity_factor = capacity_factor
        self.router_jitter = router_jitter
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd, hidden, bias, dropout) for _ in range(n_experts)])
        self.aux_loss = self.z_loss = None
        self.counts = None
        self.dropped = 0

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        n = x.shape[0]

        router_in = x
        if self.training and self.router_jitter > 0:
            router_in = x * torch.empty_like(x).uniform_(1 - self.router_jitter, 1 + self.router_jitter)
        logits = self.router(router_in).float()
        probs = F.softmax(logits, dim=-1)
        gate, idx = probs.topk(self.top_k, dim=-1)
        if self.normalize_topk:
            gate = gate / gate.sum(dim=-1, keepdim=True)

        counts = torch.bincount(idx.reshape(-1), minlength=self.n_experts)
        frac = counts.float() / (n * self.top_k)
        self.aux_loss = self.n_experts * (frac * probs.mean(dim=0)).sum()
        self.z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.counts = counts.detach()

        capacity = math.ceil(self.capacity_factor * n * self.top_k / self.n_experts) \
            if self.capacity_factor > 0 else None
        out = torch.zeros_like(x)
        dropped = 0
        for e, expert in enumerate(self.experts):
            token, slot = (idx == e).nonzero(as_tuple=True)
            if capacity is not None and token.numel() > capacity:
                dropped += token.numel() - capacity
                token, slot = token[:capacity], slot[:capacity]
            if token.numel() == 0:
                continue
            y = expert(x[token]) * gate[token, slot].unsqueeze(-1).to(x.dtype)
            out.index_add_(0, token, y)
        self.dropped = dropped
        return out.reshape(shape)
