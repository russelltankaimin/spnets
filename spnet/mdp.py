"""Exact dynamic programming over joint states (verification baseline).

Solving the induced MDP directly requires enumerating the exponentially many
joint states (Section 4); this module does exactly that, so it is only meant
for small networks.  It provides

* :class:`ExactMDPSolver` -- the optimal value ``V*_N(l)`` by backward
  induction over the finite acyclic state graph, and
* :func:`evaluate_policy_exact` -- the exact value of a (possibly randomised)
  Markov policy.

Both are used in the test-suite to certify the compositional algorithm and
the optimality of the index policy.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

from .execution import ACTIVE, COMPLETE, JointState, StaticSemantics
from .network import SPNetwork
from .policies import Policy

__all__ = ["ExactMDPSolver", "evaluate_policy_exact", "StateSpaceTooLarge"]


class StateSpaceTooLarge(RuntimeError):
    """Raised when the joint state space exceeds the configured limit."""


class _JointDP:
    """Shared memoised recursion over joint states."""

    def __init__(self, network: SPNetwork, max_states: int):
        self.network = network
        self.sem = StaticSemantics(network)
        self.betas = network.betas()
        self.max_states = max_states
        self.memo: Dict[JointState, float] = {}

    def _terminal_value(self, state: JointState) -> Optional[float]:
        st = self.sem.status(state)
        if st == ACTIVE:
            return None
        return 1.0 if st == COMPLETE else 0.0

    def q_value(self, state: JointState, task: int) -> float:
        """Value of attempting ``task`` and then continuing with :meth:`value`."""
        p = self.network.tasks[task].probs[state[0][task]]
        v_s = self.value(self.sem.after(state, task, True)) if p > 0 else 0.0
        v_f = self.value(self.sem.after(state, task, False)) if p < 1 else 0.0
        return self.betas[task] * (p * v_s + (1.0 - p) * v_f)

    def value(self, state: JointState) -> float:  # overridden
        raise NotImplementedError

    def _store(self, state: JointState, v: float) -> float:
        if len(self.memo) >= self.max_states:
            raise StateSpaceTooLarge(f"more than {self.max_states} joint states")
        self.memo[state] = v
        return v


class ExactMDPSolver(_JointDP):
    """Optimal value by brute-force backward induction (exponential in general).

    Parameters
    ----------
    network:
        A small SP network.
    max_states:
        Safety limit on the number of memoised joint states.
    """

    def __init__(self, network: SPNetwork, max_states: int = 2_000_000):
        super().__init__(network, max_states)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * network.total_states + 1000))

    def value(self, state: Optional[JointState] = None) -> float:
        """``V*`` at ``state`` (the initial state by default)."""
        state = self.sem.initial_state() if state is None else state
        if state in self.memo:
            return self.memo[state]
        term = self._terminal_value(state)
        if term is not None:
            return self._store(state, term)
        best = 0.0  # quitting is always allowed and yields 0
        for e in self.sem.frontier(state):
            best = max(best, self.q_value(state, e))
        return self._store(state, best)

    def action_values(self, state: Optional[JointState] = None) -> Dict[int, float]:
        """Q-values of all frontier tasks at ``state``."""
        state = self.sem.initial_state() if state is None else state
        return {e: self.q_value(state, e) for e in self.sem.frontier(state)}


class _PolicyEvaluator(_JointDP):
    def __init__(self, network: SPNetwork, policy: Policy, max_states: int):
        super().__init__(network, max_states)
        if not policy.markov:
            raise ValueError(f"{policy.name} is history dependent; use Monte-Carlo evaluation")
        self.policy = policy
        policy.reset()

    def value(self, state: JointState) -> float:
        if state in self.memo:
            return self.memo[state]
        term = self._terminal_value(state)
        if term is not None:
            return self._store(state, term)
        frontier = self.sem.frontier(state)
        dist = self.policy.action_distribution(frontier, state[0])
        v = sum(w * self.q_value(state, e) for e, w in dist.items())
        return self._store(state, v)


def evaluate_policy_exact(network: SPNetwork, policy: Policy, max_states: int = 2_000_000
                          ) -> float:
    """Exact expected discounted reward of a Markov policy (small networks only)."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * network.total_states + 1000))
    ev = _PolicyEvaluator(network, policy, max_states)
    return ev.value(ev.sem.initial_state())
