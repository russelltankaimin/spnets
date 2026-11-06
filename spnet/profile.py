"""Calibrated value profiles and their compositional rules (Section 4.2, App. C-F, H).

For a subcomponent ``G`` the *intrinsic calibrated value profile*
``Psi_G(s)`` (``s`` in ``[0, 1]`` is the outside option / buyout, unit exit
reward) is convex, non-decreasing, 1-Lipschitz, piecewise linear and satisfies
``s <= Psi_G(s) <= 1`` with ``Psi_G(1) = 1``.  This module represents such
profiles as ordered lists of affine pieces (:class:`Profile`) and implements

* :func:`leaf_profile` -- the profile of an atomic task (Eq. 101-102),
* :func:`par_compose`  -- ``Psi_Par(s) = 1 - int_s^1 prod_i Psi_i'(z) dz`` (Eq. 1),
  computed with a balanced product tree (Appendix H.3),
* :func:`series_compose` -- ``Psi_Series`` via transfer-map composition with a
  balanced composition tree (Appendix H.4, Eq. 119-122); with two arguments it
  is the *perspective composition* ``P_f[r](s) = r(s) f(s / r(s))`` (Eq. 83),
* :func:`smallest_fixed_points` -- task indices ``alpha = c R(alpha)`` (Eq. 88).

All functions are generic over the scalar type: they accept ``float`` or
:class:`spnet.ad.Var` inputs.  With ``Var`` inputs every arithmetic operation
is recorded, which is the tape of the n-RT algorithm.  In that mode profiles
must *not* be compacted (zero-width pieces and equal-slope neighbours carry
derivative information, Appendix H.7), hence the ``compact`` flags.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

from .ad import Scalar, value_of

__all__ = [
    "Profile", "leaf_profile", "leaf_lines", "par_compose", "series_compose",
    "smallest_fixed_points", "COMPACT_TOL",
]

#: Tolerance used when compacting float profiles.
COMPACT_TOL = 1e-13


class Profile:
    """A continuous piecewise-linear function on ``[0, 1]``.

    Piece ``j`` covers ``[breaks[j], breaks[j+1]]`` and equals
    ``slopes[j] * s + intercepts[j]`` there; ``breaks[0] = 0`` and
    ``breaks[-1] = 1``.
    """

    __slots__ = ("breaks", "slopes", "intercepts")

    def __init__(self, breaks: Sequence[Scalar], slopes: Sequence[Scalar],
                 intercepts: Sequence[Scalar]):
        if len(breaks) != len(slopes) + 1 or len(slopes) != len(intercepts) or not slopes:
            raise ValueError("inconsistent profile representation")
        self.breaks = list(breaks)
        self.slopes = list(slopes)
        self.intercepts = list(intercepts)

    # ----------------------------------------------------------- constructors
    @classmethod
    def identity(cls) -> "Profile":
        """``Psi(s) = s``: a subcomponent that can no longer be completed."""
        return cls([0.0, 1.0], [1.0], [0.0])

    @classmethod
    def constant(cls, value: float = 1.0) -> "Profile":
        """``Psi(s) = value``; ``value = 1`` is a completed subcomponent."""
        return cls([0.0, 1.0], [0.0], [value])

    # ----------------------------------------------------------- queries
    @property
    def num_pieces(self) -> int:
        return len(self.slopes)

    def piece_at(self, s: float) -> int:
        """Index of the piece containing ``s`` (right-continuous)."""
        xs = [value_of(x) for x in self.breaks[1:-1]]
        return bisect_right(xs, value_of(s))

    def __call__(self, s: Scalar) -> Scalar:
        j = self.piece_at(value_of(s))
        return self.slopes[j] * s + self.intercepts[j]

    def right_derivative(self, s: float) -> Scalar:
        return self.slopes[self.piece_at(s)]

    def is_identity(self) -> bool:
        """Whether ``Psi(0) = 0``.

        For a valid (convex, ``Psi(s) >= s``, ``Psi(1) = 1``) profile this forces
        ``Psi(s) = s``: the subcomponent can never be completed.
        """
        return value_of(self.intercepts[0]) <= 1e-15

    def to_float(self) -> "Profile":
        return Profile([value_of(x) for x in self.breaks], [value_of(a) for a in self.slopes],
                       [value_of(b) for b in self.intercepts])

    def compact(self, tol: float = COMPACT_TOL) -> "Profile":
        """Drop zero-width pieces and merge collinear neighbours (float profiles only)."""
        xs, a_s, b_s = [self.breaks[0]], [], []
        for j in range(self.num_pieces):
            x1, a, b = self.breaks[j + 1], self.slopes[j], self.intercepts[j]
            if x1 - xs[-1] <= tol and j < self.num_pieces - 1:
                continue  # zero-width piece: its neighbours meet at this point
            if a_s and abs(a - a_s[-1]) <= tol and abs(b - b_s[-1]) <= tol:
                xs[-1] = x1
                continue
            a_s.append(a)
            b_s.append(b)
            xs.append(x1)
        xs[-1] = self.breaks[-1]
        return Profile(xs, a_s, b_s)

    def pieces(self) -> List[Tuple[float, float, float, float]]:
        """``(x_left, x_right, slope, intercept)`` tuples as floats."""
        return [(value_of(self.breaks[j]), value_of(self.breaks[j + 1]),
                 value_of(self.slopes[j]), value_of(self.intercepts[j]))
                for j in range(self.num_pieces)]

    def __repr__(self) -> str:
        parts = ", ".join(f"[{x0:.4g},{x1:.4g}]: {a:.4g}s+{b:.4g}"
                          for x0, x1, a, b in self.pieces())
        return f"Profile({parts})"


# ------------------------------------------------------------------ leaf profile
def leaf_lines(beta: Scalar, probs: Sequence[float]) -> Tuple[List[Scalar], List[Scalar]]:
    """Payoff lines ``A_h + B_h s`` of attempting a task at most ``h`` times (Eq. 101).

    ``A_0 = 0, B_0 = 1, A_{h+1} = A_h + beta p(h) B_h, B_{h+1} = beta (1 - p(h)) B_h``.
    """
    A: List[Scalar] = [0.0]
    B: List[Scalar] = [1.0]
    for p in probs:
        A.append(A[-1] + beta * p * B[-1])
        B.append(beta * (1.0 - p) * B[-1])
    return A, B


def leaf_profile(beta: Scalar, probs: Sequence[float], compact: bool = True) -> Profile:
    """Intrinsic profile of an atomic task: ``max_h {A_h + B_h s}`` on ``[0, 1]`` (Eq. 102).

    The slopes ``B_h`` are non-increasing in ``h``, so the upper envelope is
    built with a monotone stack in ``O(q_e)``.  This is valid for arbitrary
    (not necessarily monotone) success-probability ladders.
    """
    A, B = leaf_lines(beta, probs)
    hull: List[Tuple[Scalar, Scalar]] = []  # lines with strictly increasing slope

    def cross(l1: Tuple[Scalar, Scalar], l2: Tuple[Scalar, Scalar]) -> Scalar:
        return (l1[0] - l2[0]) / (l2[1] - l1[1])

    for h in range(len(A) - 1, -1, -1):           # increasing slope order
        line = (A[h], B[h])
        if hull and not (line[1] > hull[-1][1]):   # equal slope: keep larger intercept
            if line[0] > hull[-1][0]:
                hull.pop()
            else:
                continue
        while len(hull) >= 2 and cross(hull[-2], line) <= cross(hull[-2], hull[-1]):
            hull.pop()
        hull.append(line)

    xs = [cross(hull[i - 1], hull[i]) for i in range(1, len(hull))]
    lo = 0
    while lo < len(xs) and xs[lo] <= 0.0:          # pieces entirely left of 0
        lo += 1
    hi = len(hull)
    while hi - 1 > lo and xs[hi - 2] >= 1.0:       # pieces entirely right of 1
        hi -= 1
    breaks: List[Scalar] = [0.0] + xs[lo:hi - 1] + [1.0]
    prof = Profile(breaks, [l[1] for l in hull[lo:hi]], [l[0] for l in hull[lo:hi]])
    return prof.compact() if compact else prof


# ------------------------------------------------------------------ segment tree
T = TypeVar("T")


class _SegmentTree(Generic[T]):
    """Ordered balanced tree with point updates and an associative ``combine``."""

    def __init__(self, leaves: Sequence[T], combine: Callable[[T, T], T], neutral: T):
        size = 1
        while size < len(leaves):
            size *= 2
        self.size, self.combine = size, combine
        self.data: List[T] = [neutral] * (2 * size)
        for i, leaf in enumerate(leaves):
            self.data[size + i] = leaf
        for i in range(size - 1, 0, -1):
            self.data[i] = combine(self.data[2 * i], self.data[2 * i + 1])

    @property
    def root(self) -> T:
        return self.data[1]

    def update(self, i: int, leaf: T) -> None:
        i += self.size
        self.data[i] = leaf
        i //= 2
        while i:
            self.data[i] = self.combine(self.data[2 * i], self.data[2 * i + 1])
            i //= 2


def _earliest(t1: Optional[Scalar], o1: int, t2: Optional[Scalar], o2: int
              ) -> Tuple[Optional[Scalar], int]:
    """Minimum of two optional event times (``None`` = no further event)."""
    if t1 is None:
        return t2, o2
    if t2 is None or t1 <= t2:
        return t1, o1
    return t2, o2


def _next_break(prof: Profile, j: int) -> Optional[Scalar]:
    return prof.breaks[j + 1] if j + 1 < prof.num_pieces else None


# ------------------------------------------------------------------ Par
def par_compose(children: Sequence[Profile], compact: bool = True) -> Profile:
    """Profile of ``Par(G_1, ..., G_m)`` from its children's profiles (Eq. 1 / Lemma D.2.2).

    The slope of the parent is the product of the active child slopes; it can
    only change at a child breakpoint.  A balanced product tree keeps the slope
    product and the earliest next child breakpoint, so each event costs
    ``O(log m)``.  The profile is then integrated backwards from ``Psi(1) = 1``.
    """
    if len(children) == 1:
        return children[0]
    ptr = [0] * len(children)

    def leaf(i: int) -> Tuple[Scalar, Optional[Scalar], int]:
        return children[i].slopes[ptr[i]], _next_break(children[i], ptr[i]), i

    def combine(l: Tuple, r: Tuple) -> Tuple:
        t, owner = _earliest(l[1], l[2], r[1], r[2])
        return l[0] * r[0], t, owner

    tree = _SegmentTree([leaf(i) for i in range(len(children))], combine, (1.0, None, -1))
    xs: List[Scalar] = [0.0]
    ds: List[Scalar] = []
    while True:
        slope, t, owner = tree.root
        ds.append(slope)
        if t is None:
            xs.append(1.0)
            break
        xs.append(t)
        ptr[owner] += 1
        tree.update(owner, leaf(owner))
    # Backward integration: y_r = 1, y_{j-1} = y_j - d_j (x_j - x_{j-1}).
    r = len(ds)
    intercepts: List[Scalar] = [0.0] * r
    y: Scalar = 1.0
    for j in range(r - 1, -1, -1):
        intercepts[j] = y - ds[j] * xs[j + 1]
        y = y - ds[j] * (xs[j + 1] - xs[j])
    prof = Profile(xs, ds, intercepts)
    return prof.compact() if compact else prof


# ------------------------------------------------------------------ Series
def series_compose(children: Sequence[Profile], compact: bool = True) -> Profile:
    """Profile of ``Series(G_1, ..., G_m)`` (``G_1`` upstream) -- Lemma D.2.3 / App. H.4.

    Each stage acts on the outside option seen by the stages upstream of it
    through the transfer map ``tau_H(s) = s / Psi_H(s)``.  On active pieces a
    consecutive block ``H`` has ``Psi_H(s) = A_H s + B_H``; blocks combine as
    ``A = A_L + B_L A_R, B = B_L B_R`` and the next event of an upstream block
    is mapped to the parent coordinate by ``t_L = eta_L B_R / (1 - eta_L A_R)``.

    With ``children = [f, r]`` this is the perspective composition
    ``P_f[r](s) = r(s) f(s / r(s))`` used for continuation profiles (Eq. 83).
    """
    if len(children) == 1:
        return children[0]
    if any(c.is_identity() for c in children):
        return Profile.identity()   # a dead stage makes the whole series dead
    ptr = [0] * len(children)

    def leaf(i: int) -> Tuple[Scalar, Scalar, Optional[Scalar], int]:
        c, j = children[i], ptr[i]
        return c.slopes[j], c.intercepts[j], _next_break(c, j), i

    def combine(l: Tuple, r: Tuple) -> Tuple:
        a_l, b_l, eta_l, o_l = l
        a_r, b_r, eta_r, o_r = r
        t_l = None if eta_l is None else eta_l * b_r / (1.0 - eta_l * a_r)
        t, owner = _earliest(t_l, o_l, eta_r, o_r)
        return a_l + b_l * a_r, b_l * b_r, t, owner

    tree = _SegmentTree([leaf(i) for i in range(len(children))], combine,
                        (0.0, 1.0, None, -1))   # neutral element: Psi = 1 (completed)
    xs: List[Scalar] = [0.0]
    slopes: List[Scalar] = []
    intercepts: List[Scalar] = []
    while True:
        a, b, t, owner = tree.root
        slopes.append(a)
        intercepts.append(b)
        if t is None or t >= 1.0:
            xs.append(1.0)
            break
        xs.append(t)
        ptr[owner] += 1
        tree.update(owner, leaf(owner))
    prof = Profile(xs, slopes, intercepts)
    return prof.compact() if compact else prof


# ------------------------------------------------------------------ fixed points
def smallest_fixed_points(cs: Sequence[float], R: Profile, tol: float = 1e-13) -> List[float]:
    """Smallest solutions of ``alpha = c R(alpha)`` for every ``c`` in ``cs`` (Eq. 88).

    ``h_c(g) = c R(g) - g`` is non-increasing in ``g`` and non-decreasing in
    ``c``, so after sorting the ``c`` values a single forward sweep through the
    pieces of ``R`` solves all equations (Corollary D.1.1):
    on a piece ``R = A + B g`` the solution is ``c A / (1 - c B)``.
    """
    order = sorted(range(len(cs)), key=lambda i: cs[i])
    out = [0.0] * len(cs)
    pieces = R.to_float().pieces()
    j = 0
    for i in order:
        c = float(cs[i])
        if c <= 0.0:
            out[i] = 0.0
            continue
        while j < len(pieces) - 1 and c * (pieces[j][2] * pieces[j][1] + pieces[j][3]) \
                - pieces[j][1] > tol:
            j += 1
        x0, x1, a, b = pieces[j]
        if c * (a * x0 + b) - x0 <= tol:
            out[i] = x0
        else:
            denom = 1.0 - c * a
            alpha = c * b / denom if denom > 0 else x1
            out[i] = min(max(alpha, x0), x1)
    return out
