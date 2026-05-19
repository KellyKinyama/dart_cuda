"""SwiGLU-MoE feed-forward — port of `lib/transformer/swiglu_moe_feed_forward.dart`.

Each expert is a SwiGLU FFN:

    y = down( silu(gate(x)) * up(x) )      where silu(z) = z * sigmoid(z)

Routing identical to `MoEFeedForward`: top-K sigmoid gating with renormalisation,
always-on shared experts, and the DeepSeek-V3 aux-loss-free routing bias.
"""

from __future__ import annotations

from typing import List, Optional

from ..tensor.gpu_tensor import Tensor
from .nn import Layer, Module


class SwiGLUExpert(Module):
    def __init__(self, embed_size: int, hidden_size: int):
        self.gate = Layer(embed_size, hidden_size, useGelu=False)
        self.up = Layer(embed_size, hidden_size, useGelu=False)
        self.down = Layer(hidden_size, embed_size, useGelu=False)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        g = self.gate.forward(x, tracker)
        u = self.up.forward(x, tracker)
        silu = g * g.sigmoid()
        gated = silu * u
        tracker.extend([silu, gated])
        return self.down.forward(gated, tracker)

    def parameters(self) -> List[Tensor]:
        return [*self.gate.parameters(), *self.up.parameters(), *self.down.parameters()]


class SwiGLUMoEFeedForward(Module):
    def __init__(
        self,
        embed_size: int,
        *,
        num_routed_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        expert_hidden_size: Optional[int] = None,
        bias_update_rate: float = 1e-3,
    ):
        assert 0 < top_k <= num_routed_experts
        self.embed_size = embed_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        # dart-torch default: embed * 8 // 3
        self.expert_hidden_size = expert_hidden_size or (embed_size * 8 // 3)
        self.bias_update_rate = bias_update_rate

        self.routed_experts: List[SwiGLUExpert] = [
            SwiGLUExpert(embed_size, self.expert_hidden_size)
            for _ in range(num_routed_experts)
        ]
        self.shared_experts: List[SwiGLUExpert] = [
            SwiGLUExpert(embed_size, self.expert_hidden_size)
            for _ in range(num_shared_experts)
        ]
        self.gate = Layer(embed_size, num_routed_experts, useGelu=False)
        self.routing_bias: List[float] = [0.0] * num_routed_experts
        self.token_load: List[int] = [0] * num_routed_experts

    # dart-torch aliases
    @property
    def numRoutedExperts(self) -> int: return self.num_routed_experts
    @property
    def numSharedExperts(self) -> int: return self.num_shared_experts
    @property
    def topK(self) -> int: return self.top_k
    @property
    def expertHiddenSize(self) -> int: return self.expert_hidden_size
    @property
    def biasUpdateRate(self) -> float: return self.bias_update_rate
    @property
    def routedExperts(self) -> List[SwiGLUExpert]: return self.routed_experts
    @property
    def sharedExperts(self) -> List[SwiGLUExpert]: return self.shared_experts
    @property
    def routingBias(self) -> List[float]: return self.routing_bias
    @property
    def tokenLoad(self) -> List[int]: return self.token_load

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        gate_logits = self.gate.forward(x, tracker)
        gate_sig = gate_logits.sigmoid()
        tracker.append(gate_sig)
        sig_vals = gate_sig.data

        routing_score = [sig_vals[i] + self.routing_bias[i]
                         for i in range(self.num_routed_experts)]
        ranked = sorted(range(self.num_routed_experts),
                        key=lambda i: routing_score[i], reverse=True)
        chosen = ranked[: self.top_k]

        chosen_scores: List[Tensor] = []
        for idx in chosen:
            mask_vals = [0.0] * self.num_routed_experts
            mask_vals[idx] = 1.0
            mask = Tensor.from_list([1, self.num_routed_experts], mask_vals)
            picked = (gate_sig * mask).sum()
            tracker.extend([mask, picked])
            chosen_scores.append(picked)

        denom = chosen_scores[0]
        for s in chosen_scores[1:]:
            denom = denom + s
        denom = denom + 1e-9
        tracker.append(denom)
        weights = [s / denom for s in chosen_scores]
        tracker.extend(weights)

        out: Optional[Tensor] = None
        for k, expert_idx in enumerate(chosen):
            self.token_load[expert_idx] += 1
            y = self.routed_experts[expert_idx].forward(x, tracker)
            weighted = y * weights[k]
            tracker.extend([y, weighted])
            out = weighted if out is None else (out + weighted)

        assert out is not None
        for s in self.shared_experts:
            sy = s.forward(x, tracker)
            out = out + sy
            tracker.extend([sy, out])
        return out

    def update_routing_bias(self, *, reset_load: bool = True) -> None:
        total = sum(self.token_load)
        if total == 0:
            return
        mean = total / self.num_routed_experts
        for i in range(self.num_routed_experts):
            err = self.token_load[i] - mean
            sign = 1.0 if err > 0 else (-1.0 if err < 0 else 0.0)
            self.routing_bias[i] -= self.bias_update_rate * sign
        if reset_load:
            for i in range(self.num_routed_experts):
                self.token_load[i] = 0

    updateRoutingBias = update_routing_bias

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for e in self.routed_experts:
            ps.extend(e.parameters())
        for e in self.shared_experts:
            ps.extend(e.parameters())
        ps.extend(self.gate.parameters())
        return ps
