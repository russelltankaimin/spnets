# spnet: adaptive planning on series-parallel networks

Reference implementation of *Adaptive Planning on Series Parallel Networks: Index
Policies and Strategic Interventions* (AISTATS 2027 submission).

It uses only the Python ≥ 3.9 standard library; there are no dependencies to install.

```bash
python3 -m spnet solve    examples/running_example.json          # V*(l) and index table
python3 -m spnet simulate examples/running_example.json -n 10000 # index policy vs baselines
python3 -m spnet gradient examples/running_example.json --check  # n-RT vs finite differences
python3 -m spnet defend   examples/running_example.json --iterations 2000   # Algorithm 1
python3 -m spnet exact    examples/running_example.json          # brute-force MDP check
python3 -m spnet generate --kind SH --tasks 450 --q 44 -o sh.json
python3 -m unittest discover -s tests                            # test-suite (~5 s)
```

## Input format

A network is a JSON object with the discount rate, the tasks (edges) and the
parse tree:

```json
{
  "lambda": 1.0,
  "tasks": {
    "A": {"length": 0.4, "p": 0.5, "q": 3, "decay": 0.8},
    "B": {"length": 0.2, "p": [0.6, 0.3]},
    "C": {"length": 0.3, "p": 0.4, "q": 5},
    "D": {"length": 0.1, "p": [0.5, 0.4]},
    "E": {"length": 0.2, "p": [0.7, 0.2]},
    "F": {"length": 0.3, "p": [0.6, 0.42, 0.29]}
  },
  "network": "Par(Series(A, C), Series(B, Par(D, E), F))"
}
```

* `length` is the duration ℓ_e of every attempt.
* `p` is either the explicit success ladder `p_e(0..q_e-1)` (its length is the
  attempt limit q_e) or a number, expanded to `p * decay**k` for `k < q`.
* `network` uses the paper's grammar (`Series`/`Ser`/`S`, `Par`/`P`/`Parallel`,
  optional `Atom(x)`). It can also be a nested list such as
  `["par", ["series", "A", "C"], "B"]` or `{"series": [...]}`.
  Nested nodes of the same type are flattened into canonical m-ary nodes.

Non-monotone ladders (violating Assumption 1) are accepted; see Appendix G below.

## Python API

```python
from spnet import SPNetwork, CompositionalSolver, IndexPolicy, Simulator, NRTGradient, RegretMatching

net = SPNetwork.from_json("examples/running_example.json")
sol = CompositionalSolver(net).solve()          # two-pass algorithm
sol.value                                       # V*(l) = Psi_N(0)
sol.index(net.task_id("C"), 2)                  # alpha(C, 2)
sol.continuations[net.task_id("A")]             # R_A, a piecewise-linear Profile

policy = IndexPolicy(sol)                       # optimal policy (Theorem 1)
Simulator(net, seed=0).estimate(policy, 10_000)

value, grad = NRTGradient(net).value_and_gradient(net.lengths())   # exact subgradient
result = RegretMatching(net, NRTGradient(net)).run(2000)            # defender allocation
```

## Where each part of the paper lives

| Paper | Code |
|---|---|
| SP networks, parse tree, canonical form (§3.1, App. B) | `spnet/network.py` — `SPNetwork`, `Atom`/`Series`/`Par`, `parse_expression` |
| Runs, frontier, eager pruning (§3.2) | `spnet/execution.py` — `RunState` (incremental), `StaticSemantics` |
| Local index c(e,k) (Eq. 12); generalised c and blocks (Eq. 90–91) | `spnet/atomic.py` |
| Calibrated profiles; leaf envelope (Eq. 101–102); Par rule (Eq. 1, App. H.3); Series rule and perspective composition (Eq. 2, 83, App. H.4) | `spnet/profile.py` — `Profile`, `leaf_profile`, `par_compose`, `series_compose` |
| Index fixed points α = c·R_e(α) (Eq. 88–89, Cor. D.1.1) | `profile.smallest_fixed_points` |
| Two-pass algorithm (Algorithms 2–3, Prop. 1) | `spnet/solver.py` — `CompositionalSolver` |
| Index policy (Thm. 1) and baselines Myopic/Greedy/Sticky/DFS/Random (App. J.2.2) | `spnet/policies.py` |
| Monte-Carlo evaluation (Table 1) | `spnet/simulation.py` |
| Joint-state MDP (the exponential baseline) | `spnet/mdp.py` |
| Removing Assumption 1 (App. G) | `spnet/monotone.py`, `atomic.attempt_blocks` |
| n-RT exact subgradient (§5, App. H) incl. tie handling (H.7) | `spnet/ad.py`, `spnet/gradient.py` — `NRTGradient` |
| Stochastic-gradient baseline SG / SG-k | `gradient.StochasticGradient` |
| Regret matching (Algorithm 1) | `spnet/defender.py` — `RegretMatching` |
| Synthetic SH / PH / MN instances | `spnet/generators.py` |

## Implementation notes

* **One code path for values and gradients.** The profile operations accept
  either floats or the reverse-mode scalar `ad.Var`. With `Var` inputs, the same
  bottom-up pass records the n-RT tape: slopes, intercepts and the moving
  breakpoints are all recorded. `ad.gradient` then performs the reverse sweep, and
  `dV*/dℓ_e = -λ β_e dV*/dβ_e` (Eq. 104). Profiles are not compacted in this mode,
  because zero-width pieces carry derivative information (App. H.7).
* **Ties (App. H.7).** Each `Var` also carries a forward-mode tangent along
  the perturbation direction `β_e(ε) = β_e(1 - ε v_e)`. Comparisons are
  lexicographic in (value, tangent), which is the first-order term of the
  paper's symbolic ε-expansion. Ties that persist to first order are broken in
  a fixed order.
* **Balanced trees.** The n-ary Par node keeps the product of active child slopes and
  the earliest next breakpoint in a segment tree. The n-ary Series node keeps
  the composed transfer-map coefficients (A, B) and the next mapped event
  `t_L = η_L B_R / (1 - η_L A_R)` (Eq. 119–122). Each event costs O(log m).
* **Non-monotone tasks.** The solver uses the generalised local index of
  Eq. 90 directly, so you don't need to build the transformed network. Within a
  block, the generalised index never drops below the block's starting index.
  The greedy policy therefore executes blocks consecutively, which is the
  recovered policy of Lemma G.2.1. `to_monotone_network` builds N~ explicitly,
  for inspection and testing.
* **Common random numbers.** In rollout i, the j-th attempt of task e succeeds
  iff `U(seed, i, e, j) < p_e(j)`, where U is a counter-based hash. All policies
  therefore see identical outcomes.

## Verification

`tests/test_spnet.py` checks the following against brute force on random small
networks, including non-monotone ladders and zero durations:

* the compositional value equals the joint-state MDP optimum;
* the exact value of the index policy equals that optimum;
* n-RT gradients match finite differences;
* the n-RT output satisfies the subgradient inequality at a tied allocation;
* the Appendix-G transformation preserves V*.

It also checks that the incremental frontier agrees with the static semantics.
Example 2 reproduces the paper's value 0.1769 (`examples/example2.json`).

Approximate timings in pure Python on synthetic instances with 450–600 controls
and Q ≈ 20K:

* index computation: 0.3–0.6 s;
* one n-RT value-and-gradient: about 5 s;
* Monte-Carlo rollouts: a few thousand per second.

## Not implemented

* The weak NP-hardness reduction for mixed ParOR/ParAND (Theorem 2) is a proof,
  not an algorithm. ParAND is not part of the input grammar.
* The 8-core parallel variant n-RT-8 is not implemented. The pure-Python tape is
  the bottleneck for large Q; a vectorised or compiled tape would be the next step.

## Note: most of the code is AI-generated (by Claude) but it is checked by me though I might overlook some things.
