"""Parsed-formula IR: a typed :class:`didactic.api.Model` wrapping
the raw :class:`formulae.matrices.DesignMatrices` so the rest of
the formula frontend operates on a typed value.

The Formula IR is the canonical *source* representation of the
formula→QVR lens.  Future versions can register it as a panproto
protocol so the lens machinery applies; for now the compiler walks
this IR directly.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx
import formulae as fo
import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame


class RandomTerm(dx.Model):
    """One random-effect group, e.g. ``(1 | g)`` or ``(x | g)``.

    Attributes
    ----------
    slope : str
        ``"Intercept"`` for ``(1 | g)``; otherwise the slope
        variable name.
    group : str
        Grouping factor name.
    """

    slope: str
    group: str


class Formula(dx.Model):
    """A parsed regression formula plus the data it was parsed
    against.

    Attributes
    ----------
    formula : str
        Original formula string (e.g. ``"y ~ x + (1 | g)"``).
    response_name : str
        Name of the response column.
    fixed_term_names : tuple[str, ...]
        Names of common (fixed-effect) terms, in design-matrix
        column order.  Includes ``"Intercept"`` when an explicit
        intercept term is present.
    random_terms : tuple[RandomTerm, ...]
        Random-effect group specifications.
    fixed_design : object
        Design matrix for the fixed terms, shape ``(N, P)``.
        Stored opaquely; the type is whatever
        :func:`formulae.design_matrices` returned.
    response_values : object
        Response column values, shape ``(N,)``.
    group_levels : Mapping[str, tuple[str, ...]]
        Canonical level ordering per grouping factor, used to
        derive deterministic plate-index tensors.
    group_indices : Mapping[str, object]
        Per-group integer index array, shape ``(N,)``.
    """

    formula: str
    response_name: str
    fixed_term_names: tuple[str, ...]
    random_terms: tuple[RandomTerm, ...]
    fixed_design: np.ndarray = dx.field(opaque=True)
    response_values: np.ndarray = dx.field(opaque=True)
    group_levels: Mapping[str, tuple[str, ...]] = dx.field(
        default_factory=dict, opaque=True
    )
    group_indices: Mapping[str, tuple[int, ...]] = dx.field(
        default_factory=dict, opaque=True
    )


def parse_formula(formula: str, data: IntoDataFrame) -> Formula:
    """Parse a brms-style formula against a dataframe and lift the
    result into a :class:`Formula` IR.

    Parameters
    ----------
    formula : str
        Formula string in brms / lme4 syntax; supports fixed terms,
        interactions, polynomial terms, and ``(slope | group)``
        random-effect groups.
    data : IntoDataFrame
        Pandas, polars, or any other Narwhals-compatible dataframe
        containing the columns referenced in the formula.

    Returns
    -------
    Formula
        Typed formula IR with design matrices, response values, and
        deterministic per-group level orderings.
    """
    nw_df = nw.from_native(data, eager_only=True)
    pandas_df = nw_df.to_pandas()
    dm = fo.design_matrices(formula, data=pandas_df)
    if dm.response is None:
        raise ValueError(
            f"parse_formula: formula {formula!r} has no response "
            f"variable on the left of `~`"
        )
    response_name = dm.response.name
    fixed_term_names: tuple[str, ...] = ()
    fixed_design: np.ndarray = np.zeros((nw_df.shape[0], 0))
    if dm.common is not None:
        fixed_term_names = tuple(dm.common.terms.keys())
        fixed_design = dm.common.design_matrix
    random_terms: list[RandomTerm] = []
    group_levels: dict[str, tuple[str, ...]] = {}
    group_indices: dict[str, tuple[int, ...]] = {}
    if dm.group is not None:
        for term_name in dm.group.terms.keys():
            # Term names in formulae are like "1|g" or "x|g".
            if "|" not in term_name:
                raise ValueError(
                    f"parse_formula: unexpected random term name "
                    f"{term_name!r}; expected `(slope | group)` syntax"
                )
            slope, group = term_name.split("|", 1)
            slope = slope.strip()
            group = group.strip()
            if slope == "1":
                slope = "Intercept"
            random_terms.append(RandomTerm(slope=slope, group=group))
            if group not in group_levels:
                levels = tuple(
                    str(v) for v in nw_df[group].drop_nulls().unique().sort().to_list()
                )
                group_levels[group] = levels
                level_index = {v: i for i, v in enumerate(levels)}
                codes = tuple(level_index[str(v)] for v in nw_df[group].to_list())
                group_indices[group] = codes
    return Formula(
        formula=formula,
        response_name=response_name,
        fixed_term_names=fixed_term_names,
        random_terms=tuple(random_terms),
        fixed_design=fixed_design,
        response_values=dm.response.design_matrix,
        group_levels=group_levels,
        group_indices=group_indices,
    )
