"""GCVaR - Exact Tail-Risk Evaluation for Multi-Factor Stochastic Optimization."""

from .core import skyband_2d, cone_transform, Fenwick
from .exact import rational, topk_exact, cvar_fractional_exact, spectral_exact
from .optimization import solve_extensive, solve_outer_approx, fractional_cvar_topmeans, instance_demands
from .oracles import full_select, full_sort, component_union, sorted_lazy
from .onion import OnionIndex, OnionIterator
from .synthetic import make_cloud
from .datasets import load_local, chronological_split

__version__ = "1.0.0"
__all__ = [
    "skyband_2d",
    "cone_transform",
    "Fenwick",
    "rational",
    "topk_exact",
    "cvar_fractional_exact",
    "spectral_exact",
    "solve_extensive",
    "solve_outer_approx",
    "fractional_cvar_topmeans",
    "instance_demands",
    "full_select",
    "full_sort",
    "component_union",
    "sorted_lazy",
    "OnionIndex",
    "OnionIterator",
    "make_cloud",
    "load_local",
    "chronological_split",
]

