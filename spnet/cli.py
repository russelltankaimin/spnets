"""Command-line interface: ``python3 -m spnet <command> network.json [options]``.

Commands
--------
solve      optimal value V*(l) and the index table alpha(e, k)
simulate   Monte-Carlo comparison of the index policy against the baselines
exact      brute-force MDP value (small networks only; for verification)
gradient   exact n-RT (sub)gradient of V*(l), optionally checked by finite differences
defend     defender allocation by regret matching (Algorithm 1)
transform  Appendix G transformation of non-monotone tasks into virtual tasks
generate   write a synthetic SH / PH / MN instance to JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from .defender import RegretMatching
from .generators import SYNTHETIC_CLASSES, synthetic_network
from .gradient import NRTGradient, StochasticGradient, finite_difference_gradient
from .mdp import ExactMDPSolver
from .monotone import to_monotone_network
from .network import SPNetwork
from .policies import baseline_policies
from .simulation import compare_policies
from .solver import CompositionalSolver


def _load(args: argparse.Namespace) -> SPNetwork:
    net = SPNetwork.from_json(args.network)
    if getattr(args, "lam", None) is not None:
        net = SPNetwork(net.tasks, net.root, args.lam, net.name)
    if getattr(args, "lengths", None):
        lengths = net.lengths()
        for item in args.lengths.split(","):
            name, val = item.split("=")
            lengths[net.task_id(name.strip())] = float(val)
        net = net.with_lengths(lengths)
    return net


def _print_table(headers: Sequence[str], rows: List[Sequence[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def cmd_solve(args: argparse.Namespace) -> None:
    net = _load(args)
    sol = CompositionalSolver(net).solve()
    if args.json:
        print(json.dumps({"value": sol.value, "indices": sol.table()}, indent=2))
        return
    print(f"network : {net.expression()}")
    print(f"tasks   : {net.num_tasks}   Q = {net.total_states}   depth = {net.depth()}")
    print(f"V*(l)   : {sol.value:.10f}\n")
    rows = [[t.name, f"{t.length:.4g}", " ".join(f"{a:.4f}" for a in sol.indices[i])]
            for i, t in enumerate(net.tasks)]
    _print_table(["task", "length", "alpha(e, k) for k = 0, 1, ..."], rows)


def cmd_simulate(args: argparse.Namespace) -> None:
    net = _load(args)
    results = compare_policies(net, baseline_policies(net), args.rollouts, args.seed)
    _print_table(["policy", "expected reward", "std. error"],
                 [[k, f"{v.mean:.4f}", f"{v.stderr:.4f}"] for k, v in results.items()])
    print(f"\nexact optimum V*(l) = {CompositionalSolver(net).solve().value:.4f}")


def cmd_exact(args: argparse.Namespace) -> None:
    net = _load(args)
    print(f"MDP optimum   : {ExactMDPSolver(net, args.max_states).value():.12f}")
    print(f"compositional : {CompositionalSolver(net).solve().value:.12f}")


def cmd_gradient(args: argparse.Namespace) -> None:
    net = _load(args)
    value, grad = NRTGradient(net).value_and_gradient(net.lengths())
    fd = finite_difference_gradient(net, net.lengths()) if args.check else None
    print(f"V*(l) = {value:.10f}\n")
    headers = ["task", "dV*/dl (n-RT)"] + (["finite diff."] if fd else [])
    rows = [[t.name, f"{g:+.8f}"] + ([f"{fd[i]:+.8f}"] if fd else [])
            for i, (t, g) in enumerate(zip(net.tasks, grad))]
    _print_table(headers, rows)


def cmd_defend(args: argparse.Namespace) -> None:
    net = _load(args)
    oracle = (NRTGradient(net) if args.oracle == "nrt"
              else StochasticGradient(net, batch_size=args.batch, seed=args.seed))

    def progress(t: int, lengths, value: float) -> None:
        if args.verbose and t % max(1, args.iterations // 10) == 0:
            print(f"  iteration {t:6d}   V*(l_t) = {value:.6f}", file=sys.stderr)

    rm = RegretMatching(net, oracle, budget=args.budget)
    uniform = rm.evaluate([1.0 / net.num_tasks] * net.num_tasks)
    res = rm.run(args.iterations, target=args.target, callback=progress)
    print(f"uniform allocation  : V* = {uniform:.6f}")
    print(f"regret matching     : V* = {res.value:.6f}  "
          f"({res.iterations} iterations, {res.seconds:.1f}s, oracle = {args.oracle})\n")
    _print_table(["task", "allocation l_e"],
                 [[t.name, f"{l:.5f}"] for t, l in zip(net.tasks, res.lengths)])
    if args.output:
        net.with_lengths(res.lengths).to_json(args.output)
        print(f"\nwrote {args.output}")


def cmd_transform(args: argparse.Namespace) -> None:
    net = _load(args)
    new, _ = to_monotone_network(net)
    text = new.to_json(args.output)
    if not args.output:
        print(text)


def cmd_generate(args: argparse.Namespace) -> None:
    net = synthetic_network(args.kind, args.tasks, args.q, seed=args.seed, lam=args.lam or 0.9)
    text = net.to_json(args.output)
    if not args.output:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spnet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_: str, network: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        if network:
            p.add_argument("network", help="network JSON file")
            p.add_argument("--lengths", help="override durations, e.g. 'A=0.2,B=0.5'")
        p.add_argument("--lambda", dest="lam", type=float, help="override the discount rate")
        p.set_defaults(func=func)
        return p

    p = add("solve", cmd_solve, "optimal value and index table")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p = add("simulate", cmd_simulate, "compare policies by Monte-Carlo")
    p.add_argument("-n", "--rollouts", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p = add("exact", cmd_exact, "brute-force MDP value (small networks)")
    p.add_argument("--max-states", type=int, default=2_000_000)
    p = add("gradient", cmd_gradient, "n-RT gradient of V*(l)")
    p.add_argument("--check", action="store_true", help="compare with finite differences")
    p = add("defend", cmd_defend, "regret-matching defender allocation")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--oracle", choices=["nrt", "sg"], default="nrt")
    p.add_argument("--batch", type=int, default=1, help="SG batch size")
    p.add_argument("--budget", type=float, default=1.0, help="total allocation 1^T l")
    p.add_argument("--target", type=float, help="stop once V* at the average is <= target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", help="write the network with the allocation as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    p = add("transform", cmd_transform, "Appendix G monotone transformation")
    p.add_argument("-o", "--output")
    p = add("generate", cmd_generate, "synthetic SH/PH/MN instance", network=False)
    p.add_argument("--kind", choices=sorted(SYNTHETIC_CLASSES), default="MN")
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--q", type=int, default=10, help="attempts per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
