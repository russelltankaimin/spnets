"""Random SP-network generators (synthetic benchmarks and tests).

:func:`random_network` draws a random canonical parse tree whose internal
nodes alternate between ``Series`` and ``Par``.  The ``series_bias`` parameter
controls the shape and yields the synthetic classes of Section 6:
series-heavy (``SH``), parallel-heavy (``PH``) and mixed-nested (``MN``).
"""

from __future__ import annotations

import random
from typing import List, Optional

from .network import Atom, Node, Par, Series, SPNetwork, Task

__all__ = ["random_network", "synthetic_network", "SYNTHETIC_CLASSES"]

#: Probability that the root is a Series node, per synthetic class.
SYNTHETIC_CLASSES = {"SH": 0.85, "PH": 0.15, "MN": 0.5}


def _random_probs(rng: random.Random, q: int, monotone: bool, p_low: float, p_high: float
                  ) -> List[float]:
    probs = [rng.uniform(p_low, p_high) for _ in range(q)]
    return sorted(probs, reverse=True) if monotone else probs


def random_network(num_tasks: int, rng: Optional[random.Random] = None, max_q: int = 3,
                   max_arity: int = 3, series_bias: float = 0.5, monotone: bool = True,
                   p_range=(0.05, 0.95), length_range=(0.05, 1.0), lam: float = 1.0,
                   name: str = "") -> SPNetwork:
    """Draw a random SP network with ``num_tasks`` tasks.

    Parameters
    ----------
    max_q:
        Attempt limits are drawn uniformly from ``1..max_q``.
    max_arity:
        Maximum number of children of an internal node.
    series_bias:
        Probability that the root (and hence every other level) is a Series node.
    monotone:
        Whether success ladders satisfy Assumption 1.
    """
    rng = rng or random.Random()
    tasks = [Task(f"t{i}", rng.uniform(*length_range),
                  tuple(_random_probs(rng, rng.randint(1, max_q), monotone, *p_range)))
             for i in range(num_tasks)]
    ids = list(range(num_tasks))
    rng.shuffle(ids)

    def build(group: List[int], series: bool) -> Node:
        if len(group) == 1:
            return Atom(group[0])
        k = rng.randint(2, min(max_arity, len(group)))
        cuts = sorted(rng.sample(range(1, len(group)), k - 1))
        parts = [group[a:b] for a, b in zip([0] + cuts, cuts + [len(group)])]
        kids = [build(p, not series) for p in parts]
        return Series(kids) if series else Par(kids)

    root = build(ids, rng.random() < series_bias)
    return SPNetwork(tasks, root, lam, name)


def synthetic_network(kind: str, num_tasks: int, q: int, seed: int = 0, lam: float = 0.9,
                      max_arity: int = 4) -> SPNetwork:
    """Synthetic ``SH`` / ``PH`` / ``MN`` instance with ``q`` attempts per task."""
    if kind not in SYNTHETIC_CLASSES:
        raise ValueError(f"kind must be one of {sorted(SYNTHETIC_CLASSES)}")
    rng = random.Random(seed)
    net = random_network(num_tasks, rng, max_q=1, max_arity=max_arity,
                         series_bias=SYNTHETIC_CLASSES[kind], lam=lam, name=kind)
    tasks = []
    for t in net.tasks:
        p0 = rng.uniform(0.2, 0.9)
        decay = rng.uniform(0.85, 0.99)
        tasks.append(Task(t.name, rng.uniform(0.0, 2.0 / num_tasks),
                          tuple(p0 * decay ** k for k in range(q))))
    return SPNetwork(tasks, net.root, lam, kind)
