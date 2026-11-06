"""Subgradient oracles for the defender's problem ``min_l V*_N(l)`` (Section 5).

* :class:`NRTGradient` -- the n-ary Reverse-Tape algorithm (Appendix H).  The
  forward pass builds every calibrated profile bottom-up while recording each
  scalar operation (slopes, intercepts *and* breakpoint locations, which move
  with ``l``); the reverse pass propagates ``dV*/d(.)`` back to every
  ``beta_e`` and finally ``dV*/dl_e = -lambda beta_e dV*/dbeta_e`` (Eq. 104).
  At regular allocations this is the exact gradient (Theorem H.5.1); at
  non-regular ones ties are resolved by the symbolic perturbation of
  Appendix H.7 (first-order version, see :mod:`spnet.ad`), giving a valid
  subgradient.
* :class:`StochasticGradient` -- the SG / SG-k baseline of the experiments:
  an unbiased Monte-Carlo estimate of ``grad V^pi(l)`` for the attacker's
  optimal index policy ``pi = pi*_l`` (an element of ``dV*`` by Danskin's
  theorem), ``dV/dl_e = E[-lambda n_e exp(-lambda T) 1{success}]``.

Both oracles expose ``value_and_gradient(lengths) -> (value, gradient)``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from . import ad
from .network import SPNetwork
from .policies import IndexPolicy
from .simulation import Simulator
from .solver import root_value

__all__ = ["GradientOracle", "NRTGradient", "StochasticGradient", "finite_difference_gradient"]


class GradientOracle(ABC):
    """Interface of (sub)gradient oracles of ``l -> V*_N(l)``."""

    exact: bool = True

    def __init__(self, network: SPNetwork):
        self.network = network

    @abstractmethod
    def value_and_gradient(self, lengths: Sequence[float]) -> Tuple[float, List[float]]:
        """Return ``(V*_N(l), g)`` with ``g`` a (sub)gradient (or an estimate of one)."""


class NRTGradient(GradientOracle):
    """Exact (sub)gradient of ``V*_N(l)`` via the n-ary Reverse-Tape algorithm.

    Parameters
    ----------
    network:
        The SP network.
    direction:
        Generic positive perturbation direction ``v`` used to break ties at
        non-regular allocations (Appendix H.7).  A random one is drawn when omitted.
    seed:
        Seed for the random direction.
    """

    def __init__(self, network: SPNetwork, direction: Optional[Sequence[float]] = None,
                 seed: int = 0):
        super().__init__(network)
        if direction is None:
            rng = random.Random(seed)
            direction = [0.5 + rng.random() for _ in range(network.num_tasks)]
        if len(direction) != network.num_tasks or min(direction) <= 0:
            raise ValueError("direction must be a positive vector with one entry per task")
        self.direction = list(direction)

    def value_and_gradient(self, lengths: Sequence[float]) -> Tuple[float, List[float]]:
        lam = self.network.lam
        # beta_e(eps) = beta_e (1 - eps v_e)  <=>  d l_e / d eps = v_e / lambda at eps = 0.
        ells = [ad.Var(l, v / lam) for l, v in zip(lengths, self.direction)]
        betas = [ad.exp(-lam * l) for l in ells]
        value = root_value(self.network, betas, compact=False)   # forward pass (tape)
        grad = ad.gradient(value, ells)                          # reverse pass
        return ad.value_of(value), grad


class StochasticGradient(GradientOracle):
    """Monte-Carlo (sub)gradient estimate with batch size ``batch_size`` (SG / SG-k)."""

    exact = False

    def __init__(self, network: SPNetwork, batch_size: int = 1, seed: int = 0):
        super().__init__(network)
        self.batch_size = batch_size
        self.seed = seed
        self._calls = 0

    def value_and_gradient(self, lengths: Sequence[float]) -> Tuple[float, List[float]]:
        net = self.network.with_lengths(lengths)
        policy = IndexPolicy.for_network(net)
        sim = Simulator(net, seed=self.seed)
        lam = net.lam
        grad = [0.0] * net.num_tasks
        total = 0.0
        for b in range(self.batch_size):
            r = sim.rollout(policy, self._calls * self.batch_size + b)
            total += r.reward
            if r.success:
                for e, n_e in enumerate(r.attempts):
                    if n_e:
                        grad[e] -= lam * n_e * r.reward
        self._calls += 1
        inv = 1.0 / self.batch_size
        return total * inv, [g * inv for g in grad]


def finite_difference_gradient(network: SPNetwork, lengths: Sequence[float], h: float = 1e-6
                               ) -> List[float]:
    """Centred finite differences of ``V*_N`` (used to verify n-RT)."""
    out = []
    for e in range(network.num_tasks):
        up, dn = list(lengths), list(lengths)
        up[e] += h
        dn[e] = max(0.0, dn[e] - h)
        width = up[e] - dn[e]
        v_up = root_value(network, network.betas(up))
        v_dn = root_value(network, network.betas(dn))
        out.append((v_up - v_dn) / width)
    return out
