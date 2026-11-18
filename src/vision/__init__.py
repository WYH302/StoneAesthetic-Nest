"""Vision-Geometric Texture Field Coupling Module."""

from .vision_parser import (
    analyze_texture_field,
    detect_slab_corners_auto,
    evaluate_placement_texture_cost,
    extract_slab_and_defects,
    parse_slab_image,
    rectify_slab_perspective,
)

__all__ = [
    "analyze_texture_field",
    "detect_slab_corners_auto",
    "evaluate_placement_texture_cost",
    "extract_slab_and_defects",
    "parse_slab_image",
    "rectify_slab_perspective",
]
