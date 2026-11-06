"""Monte-Carlo evaluation of policies (Section 6, Table 1).

Outcomes are generated with *common random numbers*: in rollout ``i`` the
``j``-th attempt of task ``e`` succeeds iff ``U_i(e, j) < p_e(j)`` for a
uniform ``U_i(e, j)`` that depends only on ``(seed, i, e, j)``.  All policies
therefore share the same random outcomes, which reduces the variance of
policy comparisons.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .execution import RunState
from .network import SPNetwork
from .policies import Policy

__all__ = ["Rollout", "Estimate", "Simulator", "compare_policies"]


@dataclass(frozen=True)
class Rollout:
    """Outcome of one run."""

    reward: float
    success: bool
    elapsed: float
    attempts: List[int]


@dataclass(frozen=True)
class Estimate:
    """Monte-Carlo estimate of a policy's expected discounted reward."""

    mean: float
    std: float
    n: int

    @property
    def stderr(self) -> float:
        return self.std / math.sqrt(self.n) if self.n > 1 else float("nan")

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.stderr:.4f} (n={self.n})"


_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64 finaliser: a fast, well-mixed 64-bit integer hash."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


class _Outcomes:
    """Counter-based common random numbers ``U(e, j)`` for one rollout.

    ``U(e, j)`` is a hash of ``(seed, rollout, e, j)``, so it does not depend on
    the order in which a policy visits tasks.
    """

    def __init__(self, seed: int, rollout: int):
        self._key = _splitmix64(_splitmix64(seed & _MASK) ^ (rollout & _MASK))

    def uniform(self, task: int, attempt: int) -> float:
        h = _splitmix64(_splitmix64(self._key ^ task) ^ (attempt * 0x632BE59BD9B4E019 & _MASK))
        return (h >> 11) * (1.0 / (1 << 53))


class Simulator:
    """Simulates runs of a policy on an SP network.

    Parameters
    ----------
    network:
        The SP network (task lengths are taken from it).
    seed:
        Base seed of the common random numbers.
    """

    def __init__(self, network: SPNetwork, seed: int = 0):
        self.network = network
        self.seed = seed
        self._state = RunState(network)

    def rollout(self, policy: Policy, index: int = 0) -> Rollout:
        """Run ``policy`` once using the outcomes of rollout number ``index``."""
        net, state = self.network, self._state
        state.reset()
        outcomes = _Outcomes(self.seed, index)
        policy.reset(random.Random(f"policy-{self.seed}-{index}"))
        while not state.is_terminal:
            frontier = state.sorted_frontier()
            task = policy.choose(frontier, state.failures)
            k = state.failures[task]
            success = outcomes.uniform(task, state.attempts[task]) < net.tasks[task].probs[k]
            state.attempt(task, success)
            policy.observe(task, success)
        reward = math.exp(-net.lam * state.elapsed) if state.succeeded else 0.0
        return Rollout(reward, state.succeeded, state.elapsed, list(state.attempts))

    def estimate(self, policy: Policy, n: int = 10_000) -> Estimate:
        """Average discounted reward over ``n`` rollouts."""
        rewards = [self.rollout(policy, i).reward for i in range(n)]
        std = statistics.pstdev(rewards) if n > 1 else 0.0
        return Estimate(statistics.fmean(rewards), std, n)


def compare_policies(network: SPNetwork, policies: Sequence[Policy], n: int = 10_000,
                     seed: int = 0) -> Dict[str, Estimate]:
    """Estimate every policy's value with common random numbers."""
    sim = Simulator(network, seed)
    return {p.name: sim.estimate(p, n) for p in policies}
