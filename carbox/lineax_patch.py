"""Monkeypatches for lineax to keep JAX tracing happy."""

from __future__ import annotations

import functools

import jax.scipy as jsp
import lineax._solver.lu as _lu


def apply_lineax_lu_transpose_patch() -> None:
    """Override LU.compute to avoid tracer->bool conversion on the trans flag."""
    if getattr(_lu, "_CARBOX_LU_PATCHED", False):
        return

    original_compute = _lu.LU.compute

    @functools.wraps(original_compute)
    def compute(self, state, vector, options):
        del options
        lu_and_piv, packed_structures, transpose = state
        del transpose  # Keep transpose static to avoid tracing issues.
        trans = 0
        vector = _lu.ravel_vector(vector, packed_structures)
        solution = jsp.linalg.lu_solve(lu_and_piv, vector, trans=trans)
        solution = _lu.unravel_solution(solution, packed_structures)
        return solution, _lu.RESULTS.successful, {}

    _lu.LU.compute = compute
    _lu._CARBOX_LU_PATCHED = True
