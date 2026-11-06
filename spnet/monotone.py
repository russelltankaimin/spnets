"""Removing Assumption 1 by grouping attempts into blocks (Appendix G).

For a task whose success probabilities are not non-increasing, consecutive
attempts are grouped into blocks by the first-drop rule (Eq. 91) and each
block ``I_{e,j}`` becomes a single-attempt *virtual task* with
``beta = a + b`` and ``p = a / (a + b)`` (Eq. 94).  The task is then replaced by
``Par(e~_1, ..., e~_B)``; the resulting network satisfies Assumption 1 and has
the same optimal value (Appendix G.1.3).  Its index policy is lifted back to
the original network by executing each selected block consecutively until
success or exhaustion (Lemma G.2.1).

Note that :class:`spnet.solver.CompositionalSolver` already handles
non-monotone tasks directly through the generalised local index; the explicit
transformation here is provided for inspection and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .atomic import AttemptBlock, attempt_blocks
from .network import Atom, Node, Par, SPNetwork, Task

__all__ = ["VirtualTask", "to_monotone_network"]


@dataclass(frozen=True)
class VirtualTask:
    """Provenance of a task in the transformed network."""

    original: int
    block: Optional[AttemptBlock]  # ``None`` for tasks that were left unchanged


def to_monotone_network(network: SPNetwork) -> Tuple[SPNetwork, List[VirtualTask]]:
    """Return the transformed network ``N~`` and, per new task, its provenance.

    The transformation depends on the current task lengths (through
    ``beta_e``), so it must be recomputed whenever ``l`` changes.
    """
    betas = network.betas()
    tasks: List[Task] = []
    provenance: List[VirtualTask] = []
    replacement: dict = {}
    for e, task in enumerate(network.tasks):
        if task.is_monotone():
            replacement[e] = [len(tasks)]
            tasks.append(task)
            provenance.append(VirtualTask(e, None))
            continue
        ids = []
        for j, block in enumerate(attempt_blocks(betas[e], task.probs)):
            ids.append(len(tasks))
            tasks.append(Task(f"{task.name}#{j + 1}", block.virtual_length(network.lam),
                              (block.virtual_prob,)))
            provenance.append(VirtualTask(e, block))
        replacement[e] = ids

    def rebuild(node: Node) -> Node:
        if isinstance(node, Atom):
            ids = replacement[node.task]
            return Atom(ids[0]) if len(ids) == 1 else Par([Atom(i) for i in ids])
        return type(node)([rebuild(c) for c in node.children])

    return SPNetwork(tasks, rebuild(network.root), network.lam, network.name), provenance
