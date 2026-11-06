"""Quantities attached to a single atomic task (Appendix D.1 and G.1).

* :func:`local_index` -- the basic index ``c(e, k) = beta p / (1 - beta (1 - p))``
  (Eq. 12): the retirement threshold of task ``e`` after ``k`` failures when
  success pays a unit reward.
* :func:`local_indices` -- ``c(e, k)`` for every failure count.  For tasks that
  violate Assumption 1 the generalised definition of Eq. 90 is used,
  ``c(e, k) = max_{k < h <= q} a_e(k, h) / (1 - b_e(k, h))``, which coincides
  with Eq. 12 under Assumption 1.
* :func:`attempt_blocks` -- the *first-drop* partition of a task's attempts into
  blocks (Eq. 91) and the parameters of the equivalent single-attempt virtual
  tasks (Eq. 94).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

__all__ = ["local_index", "local_indices", "AttemptBlock", "attempt_blocks", "is_monotone"]


def is_monotone(probs: Sequence[float]) -> bool:
    """Assumption 1: success probabilities are non-increasing in the failure count."""
    return all(a >= b for a, b in zip(probs, probs[1:]))


def local_index(beta: float, p: float) -> float:
    """``c(e, k)`` for a single attempt with success probability ``p`` (Eq. 12).

    The boundary cases ``beta = 1`` give ``1`` if ``p > 0`` and ``0`` otherwise.
    """
    if p <= 0.0:
        return 0.0
    return beta * p / (1.0 - beta * (1.0 - p))


def _block_coefficients(beta: float, probs: Sequence[float], k: int, h: int):
    """``(a_e(k, h), b_e(k, h))`` of Eq. 90: attempt consecutively from ``k`` to ``h - 1``."""
    a, b = 0.0, 1.0
    for t in range(k, h):
        a += b * beta * probs[t]
        b *= beta * (1.0 - probs[t])
    return a, b


def local_indices(beta: float, probs: Sequence[float]) -> List[float]:
    """``c(e, k)`` for ``k = 0, ..., q_e - 1``.

    Under Assumption 1 this is Eq. 12 (``O(q_e)``); otherwise the generalised
    block index of Eq. 90 is evaluated in ``O(q_e^2)``.
    """
    q = len(probs)
    if is_monotone(probs):
        return [local_index(beta, p) for p in probs]
    out: List[float] = []
    for k in range(q):
        if beta >= 1.0:
            out.append(1.0 if any(p > 0 for p in probs[k:]) else 0.0)
            continue
        best, a, b = 0.0, 0.0, 1.0
        for t in range(k, q):
            a += b * beta * probs[t]
            b *= beta * (1.0 - probs[t])
            best = max(best, a / (1.0 - b))
        out.append(best)
    return out


@dataclass(frozen=True)
class AttemptBlock:
    """A block ``[start, end)`` of consecutive attempts and its virtual task.

    ``a`` is the discounted success weight and ``b`` the discounted weight of
    exhausting the block (Eq. 90).  The equivalent single-attempt task has
    ``beta = a + b``, ``p = a / (a + b)`` and ``length = -log(a + b) / lambda``
    (Eq. 94); its local index equals ``a / (1 - b) = c(e, start)``.
    """

    start: int
    end: int
    a: float
    b: float

    @property
    def virtual_beta(self) -> float:
        return self.a + self.b

    @property
    def virtual_prob(self) -> float:
        s = self.a + self.b
        return self.a / s if s > 0 else 0.0

    def virtual_length(self, lam: float) -> float:
        s = self.a + self.b
        return -math.log(s) / lam if s > 0 else math.inf


def attempt_blocks(beta: float, probs: Sequence[float], start: int = 0) -> List[AttemptBlock]:
    """First-drop partition of the attempts ``[start, q_e)`` into blocks (Eq. 91).

    A block starting at ``k_j`` ends at the first later count whose local index
    is strictly smaller than ``c(e, k_j)``.  When ``beta = 1`` all remaining
    attempts form a single block.
    """
    q = len(probs)
    if beta >= 1.0:
        a, b = _block_coefficients(beta, probs, start, q)
        return [AttemptBlock(start, q, a, b)]
    c = local_indices(beta, probs)
    blocks: List[AttemptBlock] = []
    k = start
    while k < q:
        end = k + 1
        while end < q and c[end] >= c[k]:
            end += 1
        a, b = _block_coefficients(beta, probs, k, end)
        blocks.append(AttemptBlock(k, end, a, b))
        k = end
    return blocks
