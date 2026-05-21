"""Parsed-formula IR: a typed `didactic.api.Model` wrapping
the raw `formulae.matrices.DesignMatrices` so the rest of
the formula frontend operates on typed values.

The Formula IR is the canonical *source* representation of the
formula→QVR lens.  Future versions can register it as a panproto
protocol so the lens machinery applies; for now the compiler walks
this IR directly.

Convention
----------
Each fixed-effect *term* may produce one or more design-matrix
*columns* (a single column for ``x``, two for ``poly(x, 2)``, ``K``
for an unordered factor with ``K + 1`` levels, etc.).  R / brms
assign one coefficient per *column*; this IR follows the same
convention by exploding each term into a tuple of
`FixedColumn` records.  Multi-column terms thus produce
multiple named scalar latents downstream, with deterministic naming
``{term}_1``, ``{term}_2``, ... that mirrors R's
``poly(x, 2)1`` / ``poly(x, 2)2`` display.

Polynomial default: `formulae.design_matrices`'s ``poly``
transform is orthogonal by default (matches R's
``stats::poly``).  Raw monomials are available via
``I(x^2)`` / ``I(x**2)``.  Transforms ``log``, ``exp``, ``sqrt``,
``abs``, ``sin``, ``cos``, ``tan``, ``log10``, ``log2``,
``log1p``, ``expm1`` are wired through the formulae evaluation
namespace so users coming from R get the expected base R behaviour.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx
import formulae as fo
import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame


#: R-style transforms wired into formulae's evaluation namespace.
#: Matches base R's standard ``log`` / ``exp`` / ``sqrt`` / etc.
#: behaviour in regression formulas.
_R_TRANSFORMS = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log10": np.log10,
    "log2": np.log2,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}


class FixedColumn(dx.Model):
    """One column of the fixed-effects design matrix.

    Attributes
    ----------
    term : str
        Originating term name (e.g. ``"poly(x, 2)"`` or ``"x"``).
    name : str
        Per-column label, equal to ``term`` for single-column terms
        and ``f"{term}_{k+1}"`` (1-indexed, matching R's display) for
        multi-column terms like ``poly(x, 2)``.
    qvr_name : str
        QVR-legal identifier derived from `name` (alnum / ``_``
        only); used as the variable name in the emitted program.
    is_intercept : bool
        ``True`` for the constant-1 column.
    """

    term: str
    name: str
    qvr_name: str
    is_intercept: bool = False
    data: np.ndarray = dx.field(opaque=True)


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
        Original formula string.
    response_name : str
        Name of the response column.
    fixed_columns : tuple[FixedColumn, ...]
        One entry per design-matrix column (matches R/brms's
        one-coefficient-per-column convention).
    random_terms : tuple[RandomTerm, ...]
        Random-effect group specifications.
    response_values : np.ndarray
        Response column values, shape ``(N,)``.
    group_levels : Mapping[str, tuple[str, ...]]
        Canonical level ordering per grouping factor, used to derive
        deterministic plate-index tensors.
    group_indices : Mapping[str, tuple[int, ...]]
        Per-group integer index array, shape ``(N,)``.
    """

    formula: str
    response_name: str
    fixed_columns: tuple[FixedColumn, ...] = dx.field(
        default_factory=tuple, opaque=True
    )
    random_terms: tuple[RandomTerm, ...] = ()
    response_values: np.ndarray = dx.field(opaque=True)
    group_levels: Mapping[str, tuple[str, ...]] = dx.field(
        default_factory=dict, opaque=True
    )
    group_indices: Mapping[str, tuple[int, ...]] = dx.field(
        default_factory=dict, opaque=True
    )


class FormulaData(dx.Model):
    """The complement of a `Formula` under the
    [`quivers.formulas.compile.FormulaToQVRModule`][quivers.formulas.compile.FormulaToQVRModule] lens.

    The emitted QVR [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] carries
    the structural skeleton of the formula (which columns there are,
    keyed by their QVR-legal identifier; whether each is an
    intercept; the random-effect group / slope pairs; the family;
    the response identifier in its QVR-legal form). It does *not*
    carry:

    * the per-row data arrays (those flow through the host-data
      channel at fit time);
    * the per-column / per-group / response *original* names (the
      lens uses `_qvr_name` to normalize identifiers, which
      replaces non-alphanumeric characters with underscores and is
      therefore lossy);
    * the per-column ``term`` label (presentation, ungrouped from
      the lens forward output);
    * the original formula string (presentation: the lens emits a
      canonical AST that does not record user whitespace or
      operator-precedence choices).

    Those fields travel in the complement. ``backward(module,
    complement)`` decodes the structural fields from the Module and
    fuses them with this carrier to reproduce the original
    `Formula` verbatim.

    Attributes
    ----------
    formula : str
        Original formula string.
    response_name : str
        Original (pre-`_qvr_name`) response column name.
    response_values : np.ndarray
        Response column values, shape ``(N,)``.
    fixed_column_names : Mapping[str, tuple[str, str]]
        Per-column ``(term, name)`` keyed by ``FixedColumn.qvr_name``.
        Lets the decoder recover `FixedColumn.term` and
        `FixedColumn.name` from the qvr-name surfaced in the
        Module's latent declarations.
    fixed_column_data : Mapping[str, np.ndarray]
        Per-row predictor values, keyed by ``FixedColumn.qvr_name``.
    group_original_names : Mapping[str, str]
        Per-group ``qvr_name → original group name``.
    group_levels : Mapping[str, tuple[str, ...]]
        Canonical per-group level ordering. Needed to populate
        `Formula.group_levels` from the integer-coded
        ``object G : K`` declarations the Module records.
    group_indices : Mapping[str, tuple[int, ...]]
        Per-row integer codes for each grouping factor.
    """

    formula: str = ""
    response_name: str = ""
    response_values: np.ndarray = dx.field(opaque=True)
    fixed_column_names: Mapping[str, tuple[str, str]] = dx.field(
        default_factory=dict, opaque=True
    )
    fixed_column_data: Mapping[str, np.ndarray] = dx.field(
        default_factory=dict, opaque=True
    )
    group_original_names: Mapping[str, str] = dx.field(
        default_factory=dict, opaque=True
    )
    group_levels: Mapping[str, tuple[str, ...]] = dx.field(
        default_factory=dict, opaque=True
    )
    group_indices: Mapping[str, tuple[int, ...]] = dx.field(
        default_factory=dict, opaque=True
    )


def _qvr_name(raw: str) -> str:
    """Normalize a label into a legal QVR identifier."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _explode_term(term_name: str, term, n_obs: int) -> list[FixedColumn]:
    """Explode a single formulae Term into one or more FixedColumn
    entries, matching R/brms's one-coefficient-per-column convention.
    """
    data = np.asarray(term.data)
    if data.ndim == 1:
        # Single column; treat as one coefficient.
        is_intercept = (
            getattr(term, "kind", None) == "intercept" or term_name == "Intercept"
        )
        return [
            FixedColumn(
                term=term_name,
                name=term_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(term_name)),
                is_intercept=is_intercept,
                data=data.astype(np.float64),
            )
        ]
    if data.shape[0] != n_obs:
        raise ValueError(
            f"formula_from_data: term {term_name!r} produced "
            f"{data.shape[0]} rows, expected {n_obs}"
        )
    n_cols = data.shape[1]
    if n_cols == 1:
        is_intercept = (
            getattr(term, "kind", None) == "intercept" or term_name == "Intercept"
        )
        return [
            FixedColumn(
                term=term_name,
                name=term_name,
                qvr_name=("intercept" if is_intercept else _qvr_name(term_name)),
                is_intercept=is_intercept,
                data=data[:, 0].astype(np.float64),
            )
        ]
    return [
        FixedColumn(
            term=term_name,
            name=f"{term_name}_{k + 1}",
            qvr_name=_qvr_name(f"{term_name}_{k + 1}"),
            is_intercept=False,
            data=data[:, k].astype(np.float64),
        )
        for k in range(n_cols)
    ]


def formula_from_data(
    formula: str,
    data: IntoDataFrame,
    *,
    extra_namespace: Mapping[str, object] | None = None,
) -> Formula:
    """Build a typed `Formula` IR by lifting
    `formulae.design_matrices` over a dataframe.

    This is an adapter, not a parser: the brms-style formula syntax
    is parsed by the [`formulae`](https://bambinos.github.io/formulae/)
    library; we lift its `formulae.matrices.DesignMatrices`
    result into a typed didactic record, augmented with deterministic
    per-group level orderings and integer-code arrays derived from
    the dataframe.

    The R-style numeric transforms (``log``, ``exp``, ``sqrt``,
    ``abs``, ``sin``, ``cos``, ``tan``, ``log10``, ``log2``,
    ``log1p``, ``expm1``, ``asin``, ``acos``, ``atan``, ``sinh``,
    ``cosh``, ``tanh``) are pre-loaded into the formulae evaluation
    namespace so users coming from R / brms get the expected base
    R behaviour without explicit registration.  Polynomial terms via
    ``poly(x, k)`` are orthogonal by default, matching R's
    ``stats::poly``.

    Parameters
    ----------
    formula : str
        Formula string in brms / lme4 syntax.
    data : IntoDataFrame
        Pandas, polars, or any other Narwhals-compatible dataframe.
    extra_namespace : Mapping[str, object], optional
        Additional names visible inside the formula's expression
        evaluation, merged on top of the R-style transforms.
    """
    nw_df = nw.from_native(data, eager_only=True)
    pandas_df = nw_df.to_pandas()
    namespace: dict[str, object] = dict(_R_TRANSFORMS)
    if extra_namespace:
        namespace.update(extra_namespace)
    dm = fo.design_matrices(formula, data=pandas_df, extra_namespace=namespace)
    if dm.response is None:
        raise ValueError(
            f"formula_from_data: formula {formula!r} has no response "
            f"variable on the left of `~`"
        )
    response_name = dm.response.name
    n_obs = int(pandas_df.shape[0])

    fixed_columns: list[FixedColumn] = []
    if dm.common is not None:
        for term_name, term in dm.common.terms.items():
            fixed_columns.extend(_explode_term(term_name, term, n_obs))

    random_terms: list[RandomTerm] = []
    group_levels: dict[str, tuple[str, ...]] = {}
    group_indices: dict[str, tuple[int, ...]] = {}
    if dm.group is not None:
        for term_name in dm.group.terms.keys():
            if "|" not in term_name:
                raise ValueError(
                    f"formula_from_data: unexpected random term name "
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

    response_values = (
        np.asarray(dm.response.design_matrix).reshape(-1).astype(np.float64)
    )

    return Formula(
        formula=formula,
        response_name=response_name,
        fixed_columns=tuple(fixed_columns),
        random_terms=tuple(random_terms),
        response_values=response_values,
        group_levels=group_levels,
        group_indices=group_indices,
    )
