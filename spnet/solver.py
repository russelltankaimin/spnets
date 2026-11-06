"""Compositional computation of the optimal value and all Gittins-type indices.

This implements the two-pass algorithm of Section 4.2 / Appendix F:

* **Pass 1 (bottom-up, Algorithm 2)** builds the intrinsic calibrated value
  profile ``Psi_G`` of every subcomponent.  At the root
  ``V*_N(l) = Psi_N(0)``.
* **Pass 2 (top-down, Algorithm 3)** propagates continuation profiles: ``Par``
  children share their parent's continuation, while the children of a
  ``Series`` node receive ``R_{G_j} = P_{Psi_{S_{j+1}}}[R_G]`` via a right-to-left
  accumulator.  At a leaf the continuation is ``R_e`` and the task indices are
  the smallest fixed points ``alpha(e, k) = c(e, k) R_e(alpha(e, k))``.

By Theorem 1, attempting a frontier task with the largest ``alpha(e, k_e)`` is
an optimal policy.  Tasks violating Assumption 1 are handled through the
generalised local index of Appendix G (see :mod:`spnet.atomic`), for which the
same greedy rule executes each attempt block consecutively and recovers an
optimal policy of the original network (Lemma G.2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .ad import Scalar, value_of
from .atomic import local_indices
from .network import Atom, Node, Par, Series, SPNetwork
from .profile import Profile, leaf_profile, par_compose, series_compose, smallest_fixed_points

__all__ = ["IndexSolution", "CompositionalSolver", "build_profiles", "root_value"]


def build_profiles(network: SPNetwork, betas: Sequence[Scalar], compact: bool = True
                   ) -> List[Profile]:
    """Pass 1: intrinsic profiles of all parse-tree nodes, indexed by ``node.id``.

    ``betas`` may be floats or :class:`spnet.ad.Var` (the latter records the
    n-RT tape; use ``compact=False`` in that case).
    """
    profiles: List[Optional[Profile]] = [None] * len(network.nodes)
    for node in network.nodes:  # post-order: children first
        if isinstance(node, Atom):
            task = network.tasks[node.task]
            prof = leaf_profile(betas[node.task], task.probs, compact)
        elif isinstance(node, Par):
            prof = par_compose([profiles[c.id] for c in node.children], compact)
        else:
            prof = series_compose([profiles[c.id] for c in node.children], compact)
        profiles[node.id] = prof
    return profiles  # type: ignore[return-value]


def root_value(network: SPNetwork, betas: Sequence[Scalar], compact: bool = True) -> Scalar:
    """``V*_N = Psi_N(0)`` for the given per-task discount factors."""
    root_profile = build_profiles(network, betas, compact)[network.root.id]
    return root_profile.intercepts[0]


@dataclass
class IndexSolution:
    """Output of :class:`CompositionalSolver`.

    Attributes
    ----------
    value:
        The optimal value ``V*_N(l)``.
    indices:
        ``indices[e][k] = alpha(e, k)``, the Gittins-type index of task ``e``
        after ``k`` failures.
    local:
        ``local[e][k] = c(e, k)``, the local (myopic) index.
    profiles:
        Intrinsic profile of every parse-tree node, indexed by ``node.id``.
    continuations:
        ``continuations[e] = R_e``, the continuation profile after completing ``e``.
    """

    network: SPNetwork
    value: float
    indices: List[List[float]]
    local: List[List[float]]
    profiles: List[Profile] = field(repr=False)
    continuations: List[Profile] = field(repr=False)

    def index(self, task: int, failures: int) -> float:
        """``alpha(e, k)``; exhausted tasks have index ``-inf``."""
        row = self.indices[task]
        return row[failures] if failures < len(row) else float("-inf")

    def profile_of(self, node: Node) -> Profile:
        return self.profiles[node.id]

    def table(self) -> Dict[str, List[float]]:
        """Index table keyed by task name."""
        return {t.name: list(self.indices[i]) for i, t in enumerate(self.network.tasks)}


class CompositionalSolver:
    """Exact optimal value and task indices of an SP network (Theorem 1, Prop. 1).

    Parameters
    ----------
    network:
        The SP task network (its task lengths are used unless ``lengths`` is given).
    lengths:
        Optional override of the task durations ``l``.

    Example
    -------
    >>> sol = CompositionalSolver(net).solve()
    >>> sol.value, sol.index(net.task_id("A"), 0)
    """

    def __init__(self, network: SPNetwork, lengths: Optional[Sequence[float]] = None):
        self.network = network
        self.betas = network.betas(lengths)

    def solve(self) -> IndexSolution:
        net = self.network
        profiles = build_profiles(net, self.betas, compact=True)
        continuations: List[Optional[Profile]] = [None] * net.num_tasks
        self._propagate(net.root, Profile.constant(1.0), profiles, continuations)

        local: List[List[float]] = []
        indices: List[List[float]] = []
        for e, task in enumerate(net.tasks):
            cs = local_indices(self.betas[e], task.probs)
            local.append(cs)
            indices.append(smallest_fixed_points(cs, continuations[e]))  # type: ignore[arg-type]
        return IndexSolution(net, value_of(profiles[net.root.id].intercepts[0]),
                             indices, local, profiles, continuations)  # type: ignore[arg-type]

    # Pass 2 -------------------------------------------------------------
    def _propagate(self, node: Node, incoming: Profile, profiles: List[Profile],
                   out: List[Optional[Profile]]) -> None:
        """Top-down continuation propagation (Algorithm 3), iterative-friendly recursion."""
        stack = [(node, incoming)]
        while stack:
            node, R = stack.pop()
            if isinstance(node, Atom):
                out[node.task] = R
            elif isinstance(node, Par):
                for child in node.children:          # shared by reference
                    stack.append((child, R))
            else:
                acc = R
                for i in range(len(node.children) - 1, -1, -1):
                    child = node.children[i]
                    stack.append((child, acc))
                    if i > 0:
                        acc = series_compose([profiles[child.id], acc])
