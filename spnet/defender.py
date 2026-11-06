"""Optimal defensive resource allocation by regret matching (Section 5, Appendix A).

The defender chooses task durations ``l`` from the scaled simplex
``L = {l >= 0 : 1^T l = budget}`` (the paper uses ``budget = 1``) to solve

    min_{l in L} max_pi V^pi_N(l) = min_{l in L} V*_N(l),

where the attacker best-responds with the optimal index policy.  ``V*_N`` is
convex in ``l`` (a pointwise maximum of convex functions), so the first-order
method of Algorithm 1 (regret matching, Hart & Mas-Colell) converges to the
optimum at rate ``O(1/sqrt(T))`` given (sub)gradients from a
:class:`spnet.gradient.GradientOracle`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .gradient import GradientOracle
from .network import SPNetwork
from .solver import root_value

__all__ = ["RegretMatching", "RegretMatchingResult"]


@dataclass
class RegretMatchingResult:
    """Outcome of :meth:`RegretMatching.run`.

    Attributes
    ----------
    lengths:
        The average allocation ``(1/T) sum_t l^(t)`` returned by Algorithm 1.
    value:
        Exact attacker value ``V*_N`` at the average allocation.
    iterations:
        Number of iterations performed.
    history:
        Oracle value ``V*_N(l^(t))`` (or its estimate) at every iterate.
    seconds:
        Wall-clock time.
    """

    lengths: List[float]
    value: float
    iterations: int
    history: List[float] = field(default_factory=list, repr=False)
    seconds: float = 0.0


class RegretMatching:
    """Algorithm 1: regret matching over the (scaled) simplex of allocations.

    Parameters
    ----------
    network:
        The SP network whose task durations are being allocated.
    oracle:
        A (sub)gradient oracle of ``V*_N``.
    budget:
        Total allocation ``1^T l``.
    """

    def __init__(self, network: SPNetwork, oracle: GradientOracle, budget: float = 1.0):
        self.network = network
        self.oracle = oracle
        self.budget = float(budget)

    def run(self, iterations: int, target: Optional[float] = None,
            callback: Optional[Callable[[int, List[float], float], None]] = None,
            check_every: int = 50) -> RegretMatchingResult:
        """Run up to ``iterations`` steps of regret matching.

        If ``target`` is given, stop as soon as the exact value at the running
        average allocation is at most ``target`` (checked every ``check_every``
        iterations); this is the time-to-target protocol of Section 6.
        """
        n = self.network.num_tasks
        x = [1.0 / n] * n                 # current strategy on the unit simplex
        regrets = [0.0] * n
        avg = [0.0] * n
        history: List[float] = []
        start = time.perf_counter()
        t = 0
        for t in range(1, iterations + 1):
            lengths = [self.budget * xi for xi in x]
            value, g = self.oracle.value_and_gradient(lengths)
            history.append(value)
            for e in range(n):
                avg[e] += (x[e] - avg[e]) / t
            # Loss vector g: regret of action e is <g, x> - g_e (defender minimises V).
            inner = sum(ge * xe for ge, xe in zip(g, x))
            for e in range(n):
                regrets[e] += inner - g[e]
            positive = [max(r, 0.0) for r in regrets]
            total = sum(positive)
            x = [r / total for r in positive] if total > 0 else [1.0 / n] * n
            if callback is not None:
                callback(t, lengths, value)
            if target is not None and t % check_every == 0 and self.evaluate(avg) <= target:
                break
        lengths = [self.budget * a for a in avg]
        return RegretMatchingResult(lengths, self.evaluate(avg), t, history,
                                    time.perf_counter() - start)

    def evaluate(self, strategy: Sequence[float]) -> float:
        """Exact ``V*_N`` at the allocation ``budget * strategy``."""
        lengths = [self.budget * s for s in strategy]
        return float(root_value(self.network, self.network.betas(lengths)))
