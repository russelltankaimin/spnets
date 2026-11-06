"""spnet -- adaptive planning on series-parallel task networks.

Implementation of "Adaptive Planning on Series Parallel Networks: Index
Policies and Strategic Interventions" (AISTATS 2027 submission).

Main entry points
-----------------
:class:`SPNetwork`            network model and JSON / expression input
:class:`CompositionalSolver`  exact optimal value and Gittins-type indices (Thm. 1, Prop. 1)
:class:`IndexPolicy`          the optimal index policy, plus baselines in :mod:`spnet.policies`
:class:`Simulator`            Monte-Carlo policy evaluation with common random numbers
:class:`ExactMDPSolver`       brute-force DP over joint states (verification)
:class:`NRTGradient`          exact subgradient of V*(l) via n-ary Reverse-Tape (App. H)
:class:`StochasticGradient`   Monte-Carlo gradient baseline (SG)
:class:`RegretMatching`       defender allocation by regret matching (Algorithm 1)
:func:`to_monotone_network`   Appendix G transformation removing Assumption 1
"""

from .network import Atom, NetworkFormatError, Par, Series, SPNetwork, Task, parse_expression
from .profile import Profile, leaf_profile, par_compose, series_compose
from .solver import CompositionalSolver, IndexSolution, build_profiles, root_value
from .policies import (DFSPolicy, GreedyPolicy, IndexPolicy, MyopicPolicy, Policy,
                       RandomPolicy, StickyPolicy, baseline_policies)
from .simulation import Estimate, Rollout, Simulator, compare_policies
from .mdp import ExactMDPSolver, StateSpaceTooLarge, evaluate_policy_exact
from .gradient import GradientOracle, NRTGradient, StochasticGradient, finite_difference_gradient
from .defender import RegretMatching, RegretMatchingResult
from .monotone import VirtualTask, to_monotone_network
from .execution import RunState

__version__ = "0.1.0"

__all__ = [
    "SPNetwork", "Task", "Atom", "Series", "Par", "parse_expression", "NetworkFormatError",
    "Profile", "leaf_profile", "par_compose", "series_compose",
    "CompositionalSolver", "IndexSolution", "build_profiles", "root_value",
    "Policy", "IndexPolicy", "MyopicPolicy", "GreedyPolicy", "StickyPolicy", "DFSPolicy",
    "RandomPolicy", "baseline_policies",
    "Simulator", "Estimate", "Rollout", "compare_policies",
    "ExactMDPSolver", "evaluate_policy_exact", "StateSpaceTooLarge",
    "GradientOracle", "NRTGradient", "StochasticGradient", "finite_difference_gradient",
    "RegretMatching", "RegretMatchingResult",
    "to_monotone_network", "VirtualTask", "RunState",
]
