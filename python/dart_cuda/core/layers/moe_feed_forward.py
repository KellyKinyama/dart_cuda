"""DeepSeek-style Mixture-of-Experts feed-forward layer.

Port of `lib/transformer/moe_feed_forward.dart` (dart-torch).

Features (kept faithful to the reference):
* Fine-grained routed experts (small ReLU FFNs).
* Always-on shared experts (capture common knowledge).
* Top-K sigmoid gating with renormalisation.
* Aux-loss-free routing bias (DeepSeek-V3) updated via [update_routing_bias].

Note on shapes: dart-torch processes one token at a time. We follow the same
convention here — call `forward(token, tracker)` with `token.shape == [1, embed]`
per position. The transformer block that calls us iterates over rows.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..tensor.gpu_tensor import Tensor
from .nn import Layer, Module


class Expert(Module):
    """A single small ReLU FFN expert: embed -> hidden -> embed."""

    def __init__(self, embed_size: int, hidden_size: int):
        self.up = Layer(embed_size, hidden_size, useGelu=False)
        self.down = Layer(hidden_size, embed_size, useGelu=False)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        h = self.up.forward(x, tracker)
        h_relu = h.relu()
        tracker.append(h_relu)
        return self.down.forward(h_relu, tracker)

    def parameters(self) -> List[Tensor]:
        return [*self.up.parameters(), *self.down.parameters()]


class MoEFeedForward(Module):
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
        assert 0 < top_k <= num_routed_experts, "top_k must be in (0, num_routed_experts]"
        self.embed_size = embed_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_hidden_size = expert_hidden_size or embed_size
        self.bias_update_rate = bias_update_rate

        self.routed_experts: List[Expert] = [
            Expert(embed_size, self.expert_hidden_size)
            for _ in range(num_routed_experts)
        ]
        self.shared_experts: List[Expert] = [
            Expert(embed_size, self.expert_hidden_size)
            for _ in range(num_shared_experts)
        ]
        self.gate = Layer(embed_size, num_routed_experts, useGelu=False)

        # Non-learnable routing bias (DeepSeek-V3 aux-loss-free balancing).
        self.routing_bias: List[float] = [0.0] * num_routed_experts
        self.token_load: List[int] = [0] * num_routed_experts

    # ---- dart-torch aliases -------------------------------------------------
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
    def routedExperts(self) -> List[Expert]: return self.routed_experts
    @property
    def sharedExperts(self) -> List[Expert]: return self.shared_experts
    @property
    def routingBias(self) -> List[float]: return self.routing_bias
    @property
    def tokenLoad(self) -> List[int]: return self.token_load

    # ------------------------------------------------------------------------
    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        """Forward for a single token; x shape == [1, embed_size]."""
        # 1. Gate logits + sigmoid scores.
        gate_logits = self.gate.forward(x, tracker)             # [1, E]
        gate_sig = gate_logits.sigmoid()
        tracker.append(gate_sig)
        sig_vals = gate_sig.data                                # length E

        # 2. Routing scores (sigmoid + non-grad bias) for top-K selection.
        routing_score = [sig_vals[i] + self.routing_bias[i]
                         for i in range(self.num_routed_experts)]

        # 3. Top-K selection.
        ranked = sorted(range(self.num_routed_experts),
                        key=lambda i: routing_score[i], reverse=True)
        chosen = ranked[: self.top_k]

        # 4. Differentiable renormalised weights from chosen sigmoid scores.
        #    We need per-element column slices of gate_sig as scalar tensors.
        #    Build a one-hot mask per chosen expert and reduce-sum to a scalar.
        chosen_scores: List[Tensor] = []
        for idx in chosen:
            mask_vals = [0.0] * self.num_routed_experts
            mask_vals[idx] = 1.0
            mask = Tensor.from_list([1, self.num_routed_experts], mask_vals)
            picked = (gate_sig * mask).sum()                    # scalar [1,1]
            tracker.extend([mask, picked])
            chosen_scores.append(picked)

        denom = chosen_scores[0]
        for s in chosen_scores[1:]:
            denom = denom + s
        denom = denom + 1e-9
        tracker.append(denom)
        weights = [s / denom for s in chosen_scores]
        tracker.extend(weights)

        # 5. Combine top-K routed expert outputs.
        out: Optional[Tensor] = None
        for k, expert_idx in enumerate(chosen):
            self.token_load[expert_idx] += 1
            y = self.routed_experts[expert_idx].forward(x, tracker)
            weighted = y * weights[k]
            tracker.extend([y, weighted])
            out = weighted if out is None else (out + weighted)
            if out is not None and out is not weighted:
                tracker.append(out)
        assert out is not None  # top_k >= 1

        # 6. Shared experts (always active, weight 1).
        for s in self.shared_experts:
            sy = s.forward(x, tracker)
            out = out + sy
            tracker.extend([sy, out])

        return out

    # ------------------------------------------------------------------------
    def update_routing_bias(self, *, reset_load: bool = True) -> None:
        """DeepSeek-V3 aux-loss-free load balancing nudge."""
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

    def load_balancing_loss(self) -> float:
        total = sum(self.token_load)
        if total == 0:
            return 0.0
        l = 0.0
        for i in range(self.num_routed_experts):
            f = self.token_load[i] / total
            l += f * f
        return self.num_routed_experts * l

    loadBalancingLoss = load_balancing_loss

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for e in self.routed_experts:
            ps.extend(e.parameters())
        for e in self.shared_experts:
            ps.extend(e.parameters())
        ps.extend(self.gate.parameters())
        return ps


def moe_random(seed: Optional[int] = None):
    """Convenience seed helper (parity with dart-torch's `moeRandom`)."""
    import random
    return random.Random(seed) if seed is not None else random.Random()
