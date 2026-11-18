"""Defect-Aware Count-First Irregular Nesting Solver Core (CF-SlabNest)."""

from .geometry_kernel import (
    build_part_polygon,
    check_placement_feasible,
    expand_case_items,
    polygon_area,
    transform_polygon,
)
from .mlp_ranker import CandidateRankerMLP, extract_candidate_features, load_or_init_ranker, train_mlp_ranker
from .solver import solve_baseline, solve_cf_slabnest
from .validator import validate_nesting_solution

__all__ = [
    "CandidateRankerMLP",
    "build_part_polygon",
    "check_placement_feasible",
    "expand_case_items",
    "extract_candidate_features",
    "load_or_init_ranker",
    "polygon_area",
    "solve_baseline",
    "solve_cf_slabnest",
    "train_mlp_ranker",
    "transform_polygon",
    "validate_nesting_solution",
]
