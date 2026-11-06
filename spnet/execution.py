"""Execution semantics of runs on an SP network (Section 3.2).

A *run* is a sequence of task attempts.  The *task state*
``sigma = (F, (k_e)_{e in F})`` consists of the frontier ``F`` of currently
available tasks and their failure counts.  We use *eager pruning*: tasks that
became unavailable or no longer contribute to success are removed from the
frontier.  Concretely, a parse-tree node is

* **complete** -- Atom: succeeded; Series: all children complete; Par: any child complete;
* **dead**     -- Atom: ``q_e`` failures; Series: any child dead; Par: all children dead;
* **active**   -- otherwise.

The frontier consists of the tasks reached by descending from the root into
the *first* unfinished child of every Series node and *every* active child of
every Par node (Appendix B.1).

Two implementations are provided:

* :class:`RunState` -- an incremental, mutable state used by the Monte-Carlo
  simulator; each attempt costs time proportional to the pruned subtree.
* :class:`StaticSemantics` -- a pure function from an immutable joint state to
  the frontier, used by the exact MDP solver and exact policy evaluation.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .network import Atom, Node, Par, Series, SPNetwork

__all__ = ["RunState", "StaticSemantics", "JointState", "ACTIVE", "COMPLETE", "DEAD"]

ACTIVE, COMPLETE, DEAD = 0, 1, 2

#: Immutable joint state ``(failure counts, completed flags)`` indexed by task id.
JointState = Tuple[Tuple[int, ...], Tuple[bool, ...]]


class RunState:
    """Mutable state of a single run with incremental frontier maintenance."""

    def __init__(self, network: SPNetwork):
        self.network = network
        self.reset()

    def reset(self) -> None:
        net = self.network
        self.failures: List[int] = [0] * net.num_tasks
        self.attempts: List[int] = [0] * net.num_tasks
        self.status: List[int] = [ACTIVE] * len(net.nodes)
        self._done_children: List[int] = [0] * len(net.nodes)   # Series: completed count
        self._dead_children: List[int] = [0] * len(net.nodes)   # Par: dead count
        self.frontier: Set[int] = set(_initial_frontier(net.root))
        self.elapsed = 0.0

    # ------------------------------------------------------------- queries
    @property
    def is_terminal(self) -> bool:
        return self.status[self.network.root.id] != ACTIVE

    @property
    def succeeded(self) -> bool:
        return self.status[self.network.root.id] == COMPLETE

    def sorted_frontier(self) -> List[int]:
        """Frontier tasks in left-to-right parse-tree order (deterministic)."""
        pos = self.network.leaf_position
        return sorted(self.frontier, key=pos.__getitem__)

    # ------------------------------------------------------------- dynamics
    def attempt(self, task: int, success: bool) -> None:
        """Apply the outcome of one attempt on a frontier task."""
        if task not in self.frontier:
            raise ValueError(f"task {task} is not on the frontier")
        net = self.network
        self.attempts[task] += 1
        self.elapsed += net.tasks[task].length
        atom = net.atom_of[task]
        if success:
            self.frontier.discard(task)
            self._mark(atom, COMPLETE)
        else:
            self.failures[task] += 1
            if self.failures[task] >= net.tasks[task].q:
                self.frontier.discard(task)
                self._mark(atom, DEAD)

    def _mark(self, node: Node, status: int) -> None:
        """Set ``node``'s status and propagate completion / death upwards."""
        while True:
            self.status[node.id] = status
            # Only a completed Par (siblings pruned) or a dead Series (later
            # stages pruned) can leave frontier tasks inside the subtree.
            if (status == COMPLETE and isinstance(node, Par)) or \
                    (status == DEAD and isinstance(node, Series)):
                self._prune(node)
            parent = node.parent
            if parent is None or self.status[parent.id] != ACTIVE:
                return
            if status == COMPLETE:
                if isinstance(parent, Par):
                    node, status = parent, COMPLETE
                    continue
                self._done_children[parent.id] += 1
                if self._done_children[parent.id] == len(parent.children):
                    node, status = parent, COMPLETE
                    continue
                self.frontier.update(_initial_frontier(parent.children[node.index + 1]))
                return
            # status == DEAD
            if isinstance(parent, Series):
                node, status = parent, DEAD
                continue
            self._dead_children[parent.id] += 1
            if self._dead_children[parent.id] == len(parent.children):
                node, status = parent, DEAD
                continue
            return

    def _prune(self, node: Node) -> None:
        order = self.network.leaf_order
        for pos in range(node.leaf_lo, node.leaf_hi):
            self.frontier.discard(order[pos])


def _initial_frontier(node: Node) -> List[int]:
    """Tasks available when ``node`` is entered fresh."""
    out: List[int] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Atom):
            out.append(n.task)
        elif isinstance(n, Series):
            stack.append(n.children[0])
        else:
            stack.extend(n.children)
    return out


class StaticSemantics:
    """Frontier and terminal status as pure functions of a :data:`JointState`."""

    def __init__(self, network: SPNetwork):
        self.network = network
        self.q = [t.q for t in network.tasks]

    def initial_state(self) -> JointState:
        n = self.network.num_tasks
        return (0,) * n, (False,) * n

    def status(self, state: JointState, node: Optional[Node] = None) -> int:
        node = self.network.root if node is None else node
        k, done = state
        if isinstance(node, Atom):
            if done[node.task]:
                return COMPLETE
            return DEAD if k[node.task] >= self.q[node.task] else ACTIVE
        sts = [self.status(state, c) for c in node.children]
        if isinstance(node, Series):
            if DEAD in sts:
                return DEAD
            return COMPLETE if all(s == COMPLETE for s in sts) else ACTIVE
        if COMPLETE in sts:
            return COMPLETE
        return DEAD if all(s == DEAD for s in sts) else ACTIVE

    def frontier(self, state: JointState) -> List[int]:
        """Frontier in left-to-right order (empty at terminal states)."""
        out: List[int] = []

        def visit(node: Node) -> None:
            if self.status(state, node) != ACTIVE:
                return
            if isinstance(node, Atom):
                out.append(node.task)
            elif isinstance(node, Series):
                for c in node.children:
                    if self.status(state, c) != COMPLETE:
                        visit(c)
                        return
            else:
                for c in node.children:
                    visit(c)

        visit(self.network.root)
        return out

    @staticmethod
    def after(state: JointState, task: int, success: bool) -> JointState:
        k, done = state
        if success:
            d = list(done)
            d[task] = True
            return k, tuple(d)
        kk = list(k)
        kk[task] += 1
        return tuple(kk), done
