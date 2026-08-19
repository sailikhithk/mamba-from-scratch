"""S4 layer package."""
from .hippo import bilinear_discretize, hippo_legs, zoh_discretize
from .layer import S4Layer

__all__ = [
    "S4Layer",
    "hippo_legs",
    "zoh_discretize",
    "bilinear_discretize",
]
