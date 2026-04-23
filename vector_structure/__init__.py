"""Lightweight utilities for working with structured flat vectors and block matrices in NumPy."""

from .structure import VectorStructure, simple_slice_len

__all__ = [
    "VectorStructure",
    "simple_slice_len",
]
