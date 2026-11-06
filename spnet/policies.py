"""Agent policies: the optimal index policy and the baselines of Appendix J.2.2.

Every policy chooses one task from the current frontier ``F`` using only the
observed failure counts.  The frontier is passed in left-to-right parse-tree
order, and ties are broken in favour of the earliest task in that order, which
is the fixed deterministic tie-breaking rule assumed by Theorem 1.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from .atomic import local_index
from .network import SPNetwork
from .solver import CompositionalSolver, IndexSolution

__all__ = [
    "Policy", "IndexPolicy", "MyopicPolicy", "GreedyPolicy", "StickyPolicy",
    "DFSPolicy", "RandomPolicy", "baseline_policies",
]


class Policy(ABC):
    """Base class of agent policies.

    Subclasses implement :meth:`choose`.  Stateful (history-dependent) policies
    override :meth:`reset` / :meth:`observe` and set ``markov = False``.
    """

    name: str = "policy"
    #: Whether the decision depends only on ``(frontier, failures)``.
    markov: bool = True

    def reset(self, rng: Optional[random.Random] = None) -> None:
        """Called at the start of each run."""
        self._rng = rng or random.Random()

    @abstractmethod
    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        """Return the task (an element of ``frontier``) to attempt next."""

    def observe(self, task: int, success: bool) -> None:
        """Called after every attempt with its outcome."""

    def action_distribution(self, frontier: Sequence[int], failures: Sequence[int]
                            ) -> Dict[int, float]:
        """Distribution over actions (deterministic policies put mass 1 on :meth:`choose`)."""
        return {self.choose(frontier, failures): 1.0}

    @staticmethod
    def _argmax(frontier: Sequence[int], score) -> int:
        best, best_score = frontier[0], score(frontier[0])
        for e in frontier[1:]:
            s = score(e)
            if s > best_score:
                best, best_score = e, s
        return best

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class IndexPolicy(Policy):
    """The optimal Gittins-type index policy (Theorem 1): attempt ``argmax alpha(e, k_e)``."""

    name = "Index"

    def __init__(self, solution: IndexSolution):
        self.solution = solution

    @classmethod
    def for_network(cls, network: SPNetwork, lengths: Optional[Sequence[float]] = None
                    ) -> "IndexPolicy":
        return cls(CompositionalSolver(network, lengths).solve())

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        idx = self.solution.index
        return self._argmax(frontier, lambda e: idx(e, failures[e]))


class MyopicPolicy(Policy):
    """Largest single-task index ``c(e, k) = beta p / (1 - beta (1 - p))``; ignores downstream."""

    name = "Myopic"

    def __init__(self, network: SPNetwork, lengths: Optional[Sequence[float]] = None):
        betas = network.betas(lengths)
        self._c: List[List[float]] = [[local_index(b, p) for p in t.probs]
                                      for b, t in zip(betas, network.tasks)]

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        return self._argmax(frontier, lambda e: self._c[e][failures[e]])


class GreedyPolicy(Policy):
    """Highest current success probability ``p_e(k_e)``; ignores durations and structure."""

    name = "Greedy"

    def __init__(self, network: SPNetwork):
        self._p = [t.probs for t in network.tasks]

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        return self._argmax(frontier, lambda e: self._p[e][failures[e]])


class StickyPolicy(GreedyPolicy):
    """Keep attempting the last task while it is on the frontier, otherwise act greedily."""

    name = "Sticky"
    markov = False

    def reset(self, rng: Optional[random.Random] = None) -> None:
        super().reset(rng)
        self._last: Optional[int] = None

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        last = getattr(self, "_last", None)
        if last is not None and last in frontier:
            return last
        return super().choose(frontier, failures)

    def observe(self, task: int, success: bool) -> None:
        self._last = task


class DFSPolicy(Policy):
    """First available task in the left-to-right (depth-first) order of the parse tree."""

    name = "DFS"

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        return frontier[0]


class RandomPolicy(Policy):
    """Uniformly random frontier task."""

    name = "Random"

    def choose(self, frontier: Sequence[int], failures: Sequence[int]) -> int:
        rng = getattr(self, "_rng", None) or random
        return rng.choice(list(frontier))

    def action_distribution(self, frontier: Sequence[int], failures: Sequence[int]
                            ) -> Dict[int, float]:
        w = 1.0 / len(frontier)
        return {e: w for e in frontier}


def baseline_policies(network: SPNetwork, lengths: Optional[Sequence[float]] = None
                      ) -> List[Policy]:
    """The index policy followed by all baselines used in the paper's experiments."""
    return [
        IndexPolicy.for_network(network, lengths),
        MyopicPolicy(network, lengths),
        DFSPolicy(),
        GreedyPolicy(network),
        StickyPolicy(network),
        RandomPolicy(),
    ]
