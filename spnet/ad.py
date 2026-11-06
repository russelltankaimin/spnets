"""Minimal reverse-mode automatic differentiation used by the n-RT algorithm.

The n-ary Reverse-Tape (n-RT) algorithm (Section 5 and Appendix H of the paper)
records every scalar operation performed while building the calibrated value
profiles and then traverses those operations in reverse to obtain the exact
(sub)gradient of the root value ``V*(l)`` with respect to the defender
allocation ``l``.

Rather than hand-writing a reverse rule for every primitive, the profile code
in :mod:`spnet.profile` is written *generically* over the scalar type: it works
with plain ``float`` values (fast forward evaluation) and with :class:`Var`
values (forward evaluation that records a tape).  Every :class:`Var` stores
its parents together with the local partial derivatives, which is exactly the
tape of the paper; :func:`gradient` performs the reverse sweep.

Tie handling (Appendix H.7)
---------------------------
At a non-regular allocation, several breakpoint events coincide and the
derivative depends on how the ties are resolved.  The paper resolves ties by a
common symbolic perturbation ``beta_e(eps) = beta_e (1 - eps v_e)`` for a generic
direction ``v``.  We implement the first-order version of this rule: every
:class:`Var` additionally carries a forward-mode *tangent* (its derivative along
the perturbation direction), and comparisons are lexicographic in
``(value, tangent)``.  Two quantities whose values agree (up to a relative
tolerance) are ordered by their tangents, i.e. by the leading term of their
difference in ``eps``.  Quantities that also agree to first order are treated
as identical and broken by the fixed order in which the algorithm visits them.
"""

from __future__ import annotations

import math
from itertools import count
from typing import Iterable, List, Sequence, Tuple, Union

__all__ = ["Var", "Scalar", "value_of", "exp", "log", "gradient", "VALUE_TOL"]

#: Relative tolerance under which two values are considered tied.
VALUE_TOL = 1e-11
#: Absolute tolerance under which two tangents are considered tied.
TANGENT_TOL = 1e-11

_counter = count()


class Var:
    """A scalar node of the differentiation tape.

    Attributes
    ----------
    value:
        The primal value.
    tangent:
        Directional derivative along the tie-breaking perturbation direction.
    parents:
        Tuple of ``(parent, d self / d parent)`` pairs.
    """

    __slots__ = ("value", "tangent", "parents", "order", "adjoint")

    def __init__(self, value: float, tangent: float = 0.0,
                 parents: Tuple[Tuple["Var", float], ...] = ()):
        self.value = float(value)
        self.tangent = float(tangent)
        self.parents = parents
        self.order = next(_counter)  # creation order == topological order
        self.adjoint = 0.0

    # ------------------------------------------------------------------ algebra
    def __add__(self, other: "Scalar") -> "Var":
        if isinstance(other, Var):
            return Var(self.value + other.value, self.tangent + other.tangent,
                       ((self, 1.0), (other, 1.0)))
        return Var(self.value + other, self.tangent, ((self, 1.0),))

    __radd__ = __add__

    def __sub__(self, other: "Scalar") -> "Var":
        if isinstance(other, Var):
            return Var(self.value - other.value, self.tangent - other.tangent,
                       ((self, 1.0), (other, -1.0)))
        return Var(self.value - other, self.tangent, ((self, 1.0),))

    def __rsub__(self, other: float) -> "Var":
        return Var(other - self.value, -self.tangent, ((self, -1.0),))

    def __neg__(self) -> "Var":
        return Var(-self.value, -self.tangent, ((self, -1.0),))

    def __mul__(self, other: "Scalar") -> "Var":
        if isinstance(other, Var):
            return Var(self.value * other.value,
                       self.tangent * other.value + self.value * other.tangent,
                       ((self, other.value), (other, self.value)))
        return Var(self.value * other, self.tangent * other, ((self, float(other)),))

    __rmul__ = __mul__

    def __truediv__(self, other: "Scalar") -> "Var":
        if isinstance(other, Var):
            v = self.value / other.value
            return Var(v, (self.tangent - v * other.tangent) / other.value,
                       ((self, 1.0 / other.value), (other, -v / other.value)))
        return Var(self.value / other, self.tangent / other, ((self, 1.0 / other),))

    def __rtruediv__(self, other: float) -> "Var":
        v = other / self.value
        return Var(v, -v * self.tangent / self.value, ((self, -v / self.value),))

    # -------------------------------------------------------------- comparisons
    def __lt__(self, other: "Scalar") -> bool:
        return _compare(self, other) < 0

    def __le__(self, other: "Scalar") -> bool:
        return _compare(self, other) <= 0

    def __gt__(self, other: "Scalar") -> bool:
        return _compare(self, other) > 0

    def __ge__(self, other: "Scalar") -> bool:
        return _compare(self, other) >= 0

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Var({self.value!r}, tangent={self.tangent!r})"


Scalar = Union[float, Var]


def value_of(x: Scalar) -> float:
    """Return the primal value of a float or :class:`Var`."""
    return x.value if isinstance(x, Var) else float(x)


def _tangent_of(x: Scalar) -> float:
    return x.tangent if isinstance(x, Var) else 0.0


def _compare(a: Scalar, b: Scalar) -> int:
    """Lexicographic comparison in ``(value, tangent)`` with tolerances."""
    va, vb = value_of(a), value_of(b)
    if abs(va - vb) > VALUE_TOL * (1.0 + max(abs(va), abs(vb))):
        return -1 if va < vb else 1
    dt = _tangent_of(a) - _tangent_of(b)
    if abs(dt) > TANGENT_TOL:
        return -1 if dt < 0 else 1
    return 0


def exp(x: Scalar) -> Scalar:
    """Exponential that dispatches on the scalar type."""
    if isinstance(x, Var):
        v = math.exp(x.value)
        return Var(v, v * x.tangent, ((x, v),))
    return math.exp(x)


def log(x: Scalar) -> Scalar:
    """Natural logarithm that dispatches on the scalar type."""
    if isinstance(x, Var):
        return Var(math.log(x.value), x.tangent / x.value, ((x, 1.0 / x.value),))
    return math.log(x)


def gradient(output: Scalar, inputs: Sequence[Var]) -> List[float]:
    """Reverse sweep of the tape: return ``d output / d input`` for each input.

    Nodes are processed in decreasing creation order, which is a valid
    reverse topological order because every node is created after its parents.
    """
    if not isinstance(output, Var):  # output does not depend on the inputs
        return [0.0 for _ in inputs]
    # Collect the sub-tape reachable from the output.
    seen = {id(output)}
    stack = [output]
    nodes: List[Var] = []
    while stack:
        node = stack.pop()
        node.adjoint = 0.0
        nodes.append(node)
        for parent, _ in node.parents:
            if id(parent) not in seen:
                seen.add(id(parent))
                stack.append(parent)
    for inp in inputs:
        if id(inp) not in seen:
            inp.adjoint = 0.0
    nodes.sort(key=lambda n: n.order, reverse=True)
    output.adjoint = 1.0
    for node in nodes:
        if node.adjoint == 0.0:
            continue
        for parent, local in node.parents:
            parent.adjoint += local * node.adjoint
    return [inp.adjoint for inp in inputs]


def make_inputs(values: Iterable[float], tangents: Iterable[float]) -> List[Var]:
    """Create independent input variables with the given perturbation tangents."""
    return [Var(v, t) for v, t in zip(values, tangents)]
