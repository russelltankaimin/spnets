"""Series-parallel (SP) task networks and their parse trees (Section 3).

An SP task network is described by a parse tree with grammar::

    G := Atom(e) | Series(G1, ..., Gm) | Par(G1, ..., Gn)

``Series`` requires its children to be completed in order; ``Par`` is an
OR-composition (completing any child completes the node).  Each leaf is an
atomic :class:`Task` with a duration ``length`` (``l_e``) and a ladder of
success probabilities ``p_e(0), ..., p_e(q_e - 1)``; the task becomes
unavailable after ``q_e`` failures.

Parse trees are stored in *canonical* form (Appendix B.1): unary nodes are
removed and nested nodes of the same type are flattened into one m-ary node.

Input formats
-------------
A network is most conveniently given as JSON::

    {
      "lambda": 1.0,
      "tasks": {
        "A": {"length": 0.5, "p": [0.6, 0.4, 0.2]},
        "B": {"length": 0.3, "p": 0.5, "q": 2},
        "C": {"length": 0.2, "p": 0.7, "q": 4, "decay": 0.9}
      },
      "network": "Par(Series(A, C), B)"
    }

``network`` is either an expression string in the notation of the paper
(``Series``/``Ser``/``S`` and ``Par``/``P``/``Parallel``; ``Atom(x)`` or bare
names for tasks) or a nested list such as ``["par", ["series", "A", "C"], "B"]``.
A task's ``p`` is either an explicit list (one entry per attempt) or a number,
in which case ``q`` attempts are generated as ``p * decay**k`` (``decay``
defaults to 1).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Task", "Node", "Atom", "Series", "Par", "SPNetwork",
    "parse_expression", "NetworkFormatError",
]


class NetworkFormatError(ValueError):
    """Raised when a network description is malformed."""


# --------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class Task:
    """An atomic task (an edge of the SP network).

    Parameters
    ----------
    name:
        Unique task name.
    length:
        Duration ``l_e >= 0`` of every attempt.
    probs:
        Success probabilities ``p_e(k)`` for ``k = 0, ..., q_e - 1``.
    """

    name: str
    length: float
    probs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.probs:
            raise NetworkFormatError(f"task {self.name!r} needs at least one attempt (q_e > 0)")
        if self.length < 0 or math.isnan(self.length):
            raise NetworkFormatError(f"task {self.name!r} has invalid length {self.length}")
        for p in self.probs:
            if not 0.0 <= p <= 1.0:
                raise NetworkFormatError(f"task {self.name!r} has probability {p} outside [0, 1]")

    @property
    def q(self) -> int:
        """Attempt limit ``q_e``."""
        return len(self.probs)

    def is_monotone(self) -> bool:
        """Whether the task satisfies Assumption 1 (non-increasing ``p_e(k)``)."""
        return all(a >= b for a, b in zip(self.probs, self.probs[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {"length": self.length, "p": list(self.probs)}


# ----------------------------------------------------------------- parse-tree nodes
class Node:
    """Base class of parse-tree nodes.

    After the owning :class:`SPNetwork` is built, every node carries

    * ``id``      -- position in post-order (children before parents),
    * ``parent``  -- parent node (``None`` at the root),
    * ``index``   -- position among its parent's children,
    * ``leaf_lo``/``leaf_hi`` -- the half-open range of leaf positions (in
      left-to-right order) covered by the subtree.
    """

    kind: str = "node"

    def __init__(self, children: Sequence["Node"] = ()):
        self.children: List[Node] = list(children)
        self.id: int = -1
        self.parent: Optional[Node] = None
        self.index: int = 0
        self.leaf_lo: int = 0
        self.leaf_hi: int = 0

    @property
    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.children))})"


class Atom(Node):
    """Leaf node wrapping a single task."""

    kind = "atom"

    def __init__(self, task: int):
        super().__init__(())
        self.task = task

    @property
    def is_leaf(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Atom({self.task})"


class Series(Node):
    """Ordered AND composition: children must be completed left to right."""

    kind = "series"


class Par(Node):
    """OR composition: completing any child completes the node."""

    kind = "par"


_KIND_ALIASES = {
    "series": Series, "ser": Series, "s": Series, "seq": Series,
    "par": Par, "parallel": Par, "p": Par, "or": Par,
}


# ------------------------------------------------------------------ the network
class SPNetwork:
    """A two-terminal SP task network together with its canonical parse tree.

    Parameters
    ----------
    tasks:
        The atomic tasks; task ``i`` is referenced by ``Atom(i)``.
    root:
        Root of the parse tree (any nesting; it is canonicalised).
    lam:
        Discount rate ``lambda > 0``; a successful run with elapsed time ``T``
        pays ``exp(-lambda T)``.
    """

    def __init__(self, tasks: Sequence[Task], root: Node, lam: float = 1.0, name: str = ""):
        if lam <= 0:
            raise NetworkFormatError("lambda must be positive")
        self.tasks: List[Task] = list(tasks)
        self.lam = float(lam)
        self.name = name
        self.root = _canonicalise(root)
        self.nodes: List[Node] = []          # post-order
        self.leaf_order: List[int] = []      # task ids in left-to-right order
        self.atom_of: List[Optional[Atom]] = [None] * len(self.tasks)
        self._index_tree()
        self.name_to_id = {t.name: i for i, t in enumerate(self.tasks)}

    # ----------------------------------------------------------- construction
    def _index_tree(self) -> None:
        def visit(node: Node, parent: Optional[Node], idx: int) -> None:
            node.parent, node.index = parent, idx
            node.leaf_lo = len(self.leaf_order)
            if isinstance(node, Atom):
                if not 0 <= node.task < len(self.tasks):
                    raise NetworkFormatError(f"unknown task id {node.task}")
                if self.atom_of[node.task] is not None:
                    raise NetworkFormatError(
                        f"task {self.tasks[node.task].name!r} appears more than once")
                self.atom_of[node.task] = node
                self.leaf_order.append(node.task)
            for i, child in enumerate(node.children):
                visit(child, node, i)
            node.leaf_hi = len(self.leaf_order)
            node.id = len(self.nodes)
            self.nodes.append(node)

        visit(self.root, None, 0)
        missing = [self.tasks[i].name for i, a in enumerate(self.atom_of) if a is None]
        if missing:
            raise NetworkFormatError(f"tasks not used in the network: {missing}")
        #: ``leaf_position[e]`` is the left-to-right position of task ``e``.
        self.leaf_position: List[int] = [0] * len(self.tasks)
        for pos, task in enumerate(self.leaf_order):
            self.leaf_position[task] = pos

    # --------------------------------------------------------------- accessors
    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def total_states(self) -> int:
        """``Q = sum_e q_e``, the total number of task states."""
        return sum(t.q for t in self.tasks)

    def lengths(self) -> List[float]:
        return [t.length for t in self.tasks]

    def betas(self, lengths: Optional[Sequence[float]] = None) -> List[float]:
        """Per-attempt discount factors ``beta_e = exp(-lambda l_e)``."""
        ls = self.lengths() if lengths is None else lengths
        return [math.exp(-self.lam * l) for l in ls]

    def with_lengths(self, lengths: Sequence[float]) -> "SPNetwork":
        """Return a copy of the network with new task durations ``l``."""
        if len(lengths) != self.num_tasks:
            raise ValueError("one length per task is required")
        tasks = [replace(t, length=float(l)) for t, l in zip(self.tasks, lengths)]
        return SPNetwork(tasks, _clone(self.root), self.lam, self.name)

    def task_id(self, name: str) -> int:
        return self.name_to_id[name]

    def depth(self) -> int:
        def d(n: Node) -> int:
            return 0 if n.is_leaf else 1 + max(d(c) for c in n.children)
        return d(self.root)

    def expression(self, node: Optional[Node] = None) -> str:
        """Render (a subtree of) the parse tree in the paper's notation."""
        node = self.root if node is None else node
        if isinstance(node, Atom):
            return self.tasks[node.task].name
        head = "Series" if isinstance(node, Series) else "Par"
        return f"{head}({', '.join(self.expression(c) for c in node.children)})"

    def __repr__(self) -> str:
        return f"SPNetwork({self.expression()}, lambda={self.lam})"

    # --------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SPNetwork":
        """Build a network from the JSON-like dictionary described in the module docstring."""
        if "tasks" not in data or "network" not in data:
            raise NetworkFormatError("a network description needs 'tasks' and 'network'")
        tasks: List[Task] = []
        for name, spec in _iter_task_specs(data["tasks"]):
            tasks.append(_task_from_spec(name, spec))
        names = {t.name: i for i, t in enumerate(tasks)}
        if len(names) != len(tasks):
            raise NetworkFormatError("duplicate task names")
        structure = data["network"]
        if isinstance(structure, str):
            root = parse_expression(structure, names)
        else:
            root = _tree_from_nested(structure, names)
        return cls(tasks, root, float(data.get("lambda", 1.0)), str(data.get("name", "")))

    @classmethod
    def from_json(cls, path_or_text: str) -> "SPNetwork":
        """Load a network from a JSON file path or a JSON string."""
        text = path_or_text
        if not path_or_text.lstrip().startswith("{"):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "lambda": self.lam,
            "tasks": {t.name: t.to_dict() for t in self.tasks},
            "network": self.expression(),
        }
        if self.name:
            out["name"] = self.name
        return out

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text


# ------------------------------------------------------------------ helpers
def _iter_task_specs(spec: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(spec, dict):
        yield from spec.items()
    elif isinstance(spec, list):
        for item in spec:
            if not isinstance(item, dict) or "name" not in item:
                raise NetworkFormatError("task lists must contain objects with a 'name'")
            yield item["name"], item
    else:
        raise NetworkFormatError("'tasks' must be an object or a list")


def _task_from_spec(name: str, spec: Any) -> Task:
    if not isinstance(spec, dict):
        raise NetworkFormatError(f"task {name!r}: specification must be an object")
    length = float(spec.get("length", spec.get("l", 1.0)))
    p = spec.get("p")
    if p is None:
        raise NetworkFormatError(f"task {name!r}: missing success probabilities 'p'")
    if isinstance(p, (int, float)):
        q = int(spec.get("q", 1))
        decay = float(spec.get("decay", 1.0))
        probs = tuple(float(p) * decay ** k for k in range(q))
    else:
        probs = tuple(float(x) for x in p)
        if "q" in spec and int(spec["q"]) != len(probs):
            raise NetworkFormatError(f"task {name!r}: 'q' disagrees with len(p)")
    return Task(str(name), length, probs)


def _tree_from_nested(obj: Any, names: Dict[str, int]) -> Node:
    if isinstance(obj, str):
        if obj not in names:
            raise NetworkFormatError(f"unknown task {obj!r}")
        return Atom(names[obj])
    if isinstance(obj, dict) and len(obj) == 1:
        (kind, children), = obj.items()
        obj = [kind] + list(children)
    if isinstance(obj, list) and obj and isinstance(obj[0], str):
        kind = obj[0].lower()
        if kind == "atom" and len(obj) == 2:
            return _tree_from_nested(obj[1], names)
        if kind in _KIND_ALIASES and len(obj) >= 2:
            return _KIND_ALIASES[kind]([_tree_from_nested(c, names) for c in obj[1:]])
    raise NetworkFormatError(f"cannot interpret network fragment {obj!r}")


_TOKEN = re.compile(r"\s*(?:([A-Za-z_][\w.\-]*)|(.))")


def parse_expression(text: str, names: Dict[str, int]) -> Node:
    """Parse ``Par(Series(A, C), Series(B, Par(D, E), F))``-style expressions."""
    tokens: List[str] = []
    for ident, sym in _TOKEN.findall(text):
        if ident:
            tokens.append(ident)
        elif sym.strip():
            if sym not in "(),":
                raise NetworkFormatError(f"unexpected character {sym!r} in {text!r}")
            tokens.append(sym)
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def take(expected: Optional[str] = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise NetworkFormatError(f"unexpected end of expression {text!r}")
        tok = tokens[pos]
        if expected is not None and tok != expected:
            raise NetworkFormatError(f"expected {expected!r} but found {tok!r} in {text!r}")
        pos += 1
        return tok

    def parse() -> Node:
        tok = take()
        if peek() == "(":
            take("(")
            args = [parse()]
            while peek() == ",":
                take(",")
                args.append(parse())
            take(")")
            kind = tok.lower()
            if kind == "atom":
                if len(args) != 1:
                    raise NetworkFormatError("Atom takes exactly one task")
                return args[0]
            if kind not in _KIND_ALIASES:
                raise NetworkFormatError(f"unknown composition {tok!r}")
            return _KIND_ALIASES[kind](args)
        if tok not in names:
            raise NetworkFormatError(f"unknown task {tok!r}")
        return Atom(names[tok])

    root = parse()
    if pos != len(tokens):
        raise NetworkFormatError(f"trailing tokens in {text!r}")
    return root


def _canonicalise(node: Node) -> Node:
    """Remove unary nodes and flatten nested nodes of the same type."""
    if isinstance(node, Atom):
        return Atom(node.task)
    kids: List[Node] = []
    for child in node.children:
        c = _canonicalise(child)
        if type(c) is type(node):
            kids.extend(c.children)
        else:
            kids.append(c)
    if not kids:
        raise NetworkFormatError("composition nodes need at least one child")
    if len(kids) == 1:
        return kids[0]
    return type(node)(kids)


def _clone(node: Node) -> Node:
    if isinstance(node, Atom):
        return Atom(node.task)
    return type(node)([_clone(c) for c in node.children])
