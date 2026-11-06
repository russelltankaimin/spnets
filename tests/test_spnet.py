"""Test-suite for spnet (run with ``python3 -m unittest discover -s tests`` or pytest).

The central checks certify the paper's claims against brute force on random
small networks:

* ``Psi_N(0)`` from the compositional algorithm equals the MDP optimum;
* the index policy attains the MDP optimum (Theorem 1), also for
  non-monotone success ladders (Appendix G);
* n-RT gradients agree with finite differences at regular allocations;
* the Appendix G transformation preserves the optimal value.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spnet import (CompositionalSolver, DFSPolicy, ExactMDPSolver, GreedyPolicy, IndexPolicy,
                   MyopicPolicy, NRTGradient, RandomPolicy, RegretMatching, RunState,
                   Simulator, SPNetwork, StochasticGradient, evaluate_policy_exact,
                   finite_difference_gradient, leaf_profile, par_compose, root_value,
                   series_compose, to_monotone_network)
from spnet.atomic import attempt_blocks, local_indices
from spnet.execution import StaticSemantics
from spnet.generators import random_network, synthetic_network
from spnet.network import NetworkFormatError

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def small_networks(count=60, seed=1, **kw):
    rng = random.Random(seed)
    for _ in range(count):
        yield random_network(rng.randint(1, 6), rng, max_q=3, **kw)


class TestInput(unittest.TestCase):
    def test_expression_and_nested_forms_agree(self):
        tasks = {n: {"length": 0.1, "p": 0.5, "q": 2} for n in "ABCDEF"}
        a = SPNetwork.from_dict({"tasks": tasks,
                                 "network": "Par(Series(A, C), Series(B, Par(D, E), F))"})
        b = SPNetwork.from_dict({"tasks": tasks, "network":
                                 ["par", ["series", "A", "C"],
                                  {"series": ["B", ["par", "D", "E"], "F"]}]})
        self.assertEqual(a.expression(), b.expression())

    def test_canonicalisation_flattens(self):
        tasks = {n: {"p": [0.5]} for n in "ABC"}
        net = SPNetwork.from_dict({"tasks": tasks, "network": "Series(A, Series(B, Par(C)))"})
        self.assertEqual(net.expression(), "Series(A, B, C)")

    def test_round_trip(self):
        net = SPNetwork.from_json(os.path.join(EXAMPLES, "running_example.json"))
        again = SPNetwork.from_json(net.to_json())
        self.assertEqual(net.expression(), again.expression())
        self.assertEqual([t.probs for t in net.tasks], [t.probs for t in again.tasks])

    def test_errors(self):
        tasks = {"A": {"p": [0.5]}, "B": {"p": [0.5]}}
        with self.assertRaises(NetworkFormatError):
            SPNetwork.from_dict({"tasks": tasks, "network": "Par(A, A)"})
        with self.assertRaises(NetworkFormatError):
            SPNetwork.from_dict({"tasks": tasks, "network": "Par(A)"})   # B unused
        with self.assertRaises(NetworkFormatError):
            SPNetwork.from_dict({"tasks": {"A": {"p": [1.5]}}, "network": "A"})


class TestProfiles(unittest.TestCase):
    def assert_valid(self, prof):
        self.assertAlmostEqual(prof(1.0), 1.0, places=12)
        prev = -1.0
        for x in [i / 200 for i in range(201)]:
            v = prof(x)
            self.assertGreaterEqual(v, x - 1e-12)
            self.assertLessEqual(v, 1 + 1e-12)
            self.assertGreaterEqual(v, prev - 1e-12)
            prev = v

    def test_leaf_profile_matches_recursion(self):
        rng = random.Random(0)
        for _ in range(50):
            beta = rng.uniform(0.2, 1.0)
            probs = [rng.random() for _ in range(rng.randint(1, 5))]
            prof = leaf_profile(beta, probs)
            self.assert_valid(prof)
            for s in [0.0, 0.1, 0.37, 0.8, 1.0]:
                v = s                                  # Eq. 13 backwards
                for p in reversed(probs):
                    v = max(s, beta * (p + (1 - p) * v))
                self.assertAlmostEqual(prof(s), v, places=12)

    def test_compositions_valid_and_series_identity(self):
        rng = random.Random(3)
        leaves = [leaf_profile(rng.uniform(0.3, 1), [rng.random() for _ in range(3)])
                  for _ in range(4)]
        self.assert_valid(par_compose(leaves))
        ser = series_compose(leaves)
        self.assert_valid(ser)
        # Binary series rule (Corollary D.2.1) checked pointwise.
        two = series_compose(leaves[:2])
        for s in [0.0, 0.2, 0.5, 0.9]:
            r = leaves[1](s)
            self.assertAlmostEqual(two(s), r * leaves[0](s / r), places=12)
        # Associativity of the n-ary sweep.
        nested = series_compose([leaves[0], series_compose(leaves[1:])])
        for s in [0.0, 0.3, 0.7]:
            self.assertAlmostEqual(ser(s), nested(s), places=12)


class TestOptimality(unittest.TestCase):
    def test_value_matches_mdp(self):
        for net in small_networks():
            self.assertAlmostEqual(CompositionalSolver(net).solve().value,
                                   ExactMDPSolver(net).value(), places=10, msg=repr(net))

    def test_index_policy_is_optimal(self):
        for net in small_networks(seed=2):
            opt = ExactMDPSolver(net).value()
            self.assertAlmostEqual(evaluate_policy_exact(net, IndexPolicy.for_network(net)),
                                   opt, places=10, msg=repr(net))

    def test_non_monotone_ladders(self):
        for net in small_networks(count=60, seed=4, monotone=False):
            opt = ExactMDPSolver(net).value()
            self.assertAlmostEqual(CompositionalSolver(net).solve().value, opt, places=10)
            self.assertAlmostEqual(evaluate_policy_exact(net, IndexPolicy.for_network(net)),
                                   opt, places=10, msg=repr(net))

    def test_zero_lengths(self):
        for net in small_networks(count=20, seed=5):
            lengths = [0.0 if i % 2 == 0 else t.length for i, t in enumerate(net.tasks)]
            net = net.with_lengths(lengths)
            opt = ExactMDPSolver(net).value()
            self.assertAlmostEqual(CompositionalSolver(net).solve().value, opt, places=10)
            self.assertAlmostEqual(evaluate_policy_exact(net, IndexPolicy.for_network(net)),
                                   opt, places=10)

    def test_baselines_never_beat_index(self):
        for net in small_networks(count=20, seed=6):
            opt = ExactMDPSolver(net).value()
            for pol in (MyopicPolicy(net), GreedyPolicy(net), DFSPolicy(), RandomPolicy()):
                self.assertLessEqual(evaluate_policy_exact(net, pol), opt + 1e-12)

    def test_example2(self):
        net = SPNetwork.from_json(os.path.join(EXAMPLES, "example2.json"))
        self.assertAlmostEqual(CompositionalSolver(net).solve().value, 0.1769, places=4)
        a_only = math.exp(-2.5)                       # the naive policy always picks A
        self.assertAlmostEqual(a_only, 0.0821, places=4)

    def test_index_locality_atomic(self):
        # A single task finishing the objective has index c(e, k) (Lemma D.1.1).
        net = SPNetwork.from_dict({"lambda": 0.7, "tasks": {"A": {"length": 0.5,
                                   "p": [0.6, 0.4, 0.1]}}, "network": "A"})
        sol = CompositionalSolver(net).solve()
        self.assertEqual(sol.indices[0], sol.local[0])


class TestMonotoneTransform(unittest.TestCase):
    def test_transform_preserves_value(self):
        for net in small_networks(count=40, seed=7, monotone=False):
            new, prov = to_monotone_network(net)
            self.assertTrue(all(t.is_monotone() for t in new.tasks))
            self.assertAlmostEqual(CompositionalSolver(new).solve().value,
                                   ExactMDPSolver(net).value(), places=10)
            self.assertLessEqual(new.total_states, net.total_states)

    def test_blocks_and_generalised_index(self):
        beta, probs = 0.9, [0.2, 0.6, 0.5, 0.1, 0.4, 0.05]
        blocks = attempt_blocks(beta, probs)
        cs = local_indices(beta, probs)
        self.assertEqual(blocks[0].start, 0)
        self.assertEqual(blocks[-1].end, len(probs))
        for blk in blocks:                            # Lemma G.1.1
            self.assertAlmostEqual(blk.a / (1 - blk.b), cs[blk.start], places=12)
            for t in range(blk.start + 1, blk.end):
                self.assertGreaterEqual(cs[t], cs[blk.start] - 1e-15)


class TestExecution(unittest.TestCase):
    def test_incremental_frontier_matches_static(self):
        rng = random.Random(11)
        for net in small_networks(count=40, seed=8):
            sem, state = StaticSemantics(net), RunState(net)
            joint = sem.initial_state()
            while not state.is_terminal:
                self.assertEqual(state.sorted_frontier(), sem.frontier(joint))
                e = rng.choice(state.sorted_frontier())
                ok = rng.random() < 0.4
                state.attempt(e, ok)
                joint = sem.after(joint, e, ok)
            self.assertEqual(sem.frontier(joint), [])

    def test_simulation_matches_exact(self):
        net = SPNetwork.from_json(os.path.join(EXAMPLES, "running_example.json"))
        pol = IndexPolicy.for_network(net)
        est = Simulator(net, seed=3).estimate(pol, 20000)
        self.assertLess(abs(est.mean - evaluate_policy_exact(net, pol)), 4 * est.stderr)


class TestGradients(unittest.TestCase):
    def test_nrt_matches_finite_differences(self):
        rng = random.Random(12)
        for net in small_networks(count=30, seed=9):
            lengths = [rng.uniform(0.05, 1.0) for _ in net.tasks]
            value, grad = NRTGradient(net).value_and_gradient(lengths)
            self.assertAlmostEqual(value, root_value(net, net.betas(lengths)), places=12)
            fd = finite_difference_gradient(net, lengths, h=1e-6)
            for g, f in zip(grad, fd):
                self.assertAlmostEqual(g, f, delta=1e-5 * (1 + abs(f)))

    def test_nrt_large_network(self):
        net = synthetic_network("MN", 40, 6, seed=1)
        lengths = net.lengths()
        _, grad = NRTGradient(net).value_and_gradient(lengths)
        fd = finite_difference_gradient(net, lengths, h=1e-7)
        for g, f in zip(grad, fd):
            self.assertAlmostEqual(g, f, delta=1e-5 * (1 + abs(f)))

    def test_subgradient_inequality_at_tie(self):
        # Two identical parallel tasks: a non-regular allocation (Appendix H.7).
        net = SPNetwork.from_dict({"tasks": {"A": {"p": [0.5, 0.3]}, "B": {"p": [0.5, 0.3]}},
                                   "network": "Par(A, B)"})
        ell = [0.5, 0.5]
        v, g = NRTGradient(net).value_and_gradient(ell)
        rng = random.Random(0)
        for _ in range(200):
            y = [rng.uniform(0, 1.5), rng.uniform(0, 1.5)]
            vy = root_value(net, net.betas(y))
            self.assertGreaterEqual(vy, v + sum(gi * (yi - li) for gi, yi, li in
                                                zip(g, y, ell)) - 1e-10)

    def test_stochastic_gradient_unbiased(self):
        net = SPNetwork.from_json(os.path.join(EXAMPLES, "running_example.json"))
        lengths = net.lengths()
        _, exact = NRTGradient(net).value_and_gradient(lengths)
        _, est = StochasticGradient(net, batch_size=40000, seed=1).value_and_gradient(lengths)
        for g, s in zip(exact, est):
            self.assertAlmostEqual(g, s, delta=0.02)

    def test_regret_matching_decreases_value(self):
        net = SPNetwork.from_json(os.path.join(EXAMPLES, "running_example.json"))
        rm = RegretMatching(net, NRTGradient(net))
        uniform = rm.evaluate([1 / net.num_tasks] * net.num_tasks)
        res = rm.run(300)
        self.assertAlmostEqual(sum(res.lengths), 1.0, places=9)
        self.assertLess(res.value, uniform)


if __name__ == "__main__":
    unittest.main()
