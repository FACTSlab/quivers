"""Column-level encoding utilities: dtype dispatch, missing-data
policies, and the role enum that classifies what a column is *for*
in a probabilistic program (object axis, observed site, plate index,
covariate).
"""

from __future__ import annotations

from enum import Enum

import narwhals as nw
import torch


class ColumnRole(str, Enum):
    """How a dataframe column participates in a QVR program."""

    #: The column's unique values define the cardinality of a
    #: finite-set object.
    OBJECT = "object"

    #: The column carries observed values for an ``observe`` site;
    #: dtype dictates whether the encoded tensor is a ``LongTensor``
    #: (categorical) or ``FloatTensor`` (numeric).
    OBSERVATION = "observation"

    #: The column carries per-row integer indices into a sibling
    #: ``OBJECT`` column; encoded as a ``LongTensor`` of codes.
    PLATE_INDEX = "plate_index"

    #: The column is a numeric covariate (e.g. predictor in a
    #: regression); encoded as a ``FloatTensor``.
    COVARIATE = "covariate"


class MissingPolicy(str, Enum):
    """How to handle ``NaN`` / null entries when encoding a column."""

    #: Raise if any rows have missing values for this column.
    RAISE = "raise"

    #: Drop rows that have any missing value in the encoded column.
    #: The caller is responsible for filtering the parent dataframe.
    DROP = "drop"

    #: Replace missing entries with the column's running mean
    #: (numeric) or modal value (categorical) before encoding.
    IMPUTE = "impute"

    #: Encode missing entries as ``NaN`` for numeric columns and as
    #: an explicit ``-1`` code for categorical columns; the caller
    #: must pair this with a likelihood that handles missing data.
    MASK = "mask"


def _is_numeric_dtype(dtype) -> bool:
    """True for Narwhals numeric dtypes (Int*, UInt*, Float*).

    Booleans are treated as numeric for tensor-encoding purposes.
    """
    return any(
        dtype == kind
        for kind in (
            nw.Int8,
            nw.Int16,
            nw.Int32,
            nw.Int64,
            nw.UInt8,
            nw.UInt16,
            nw.UInt32,
            nw.UInt64,
            nw.Float32,
            nw.Float64,
            nw.Boolean,
        )
    )


def encode_column(
    df: nw.DataFrame,
    column: str,
    *,
    role: ColumnRole,
    categories: tuple[str, ...] | None = None,
    missing_policy: MissingPolicy = MissingPolicy.RAISE,
) -> torch.Tensor:
    """Encode a single column into a ``torch.Tensor`` ready for
    QVR inference.

    Parameters
    ----------
    df : nw.DataFrame
        Narwhals-wrapped dataframe.
    column : str
        Column to encode.
    role : ColumnRole
        How the column participates in the program. ``PLATE_INDEX``
        and ``OBJECT`` columns require a categories tuple for
        reproducible code assignment.
    categories : tuple[str, ...] or None
        Canonical ordering of categorical values; if provided, codes
        are assigned by ``categories.index(value)``. Required for
        ``PLATE_INDEX`` and for ``OBSERVATION`` of a non-numeric
        column. ``None`` is allowed for numeric ``OBSERVATION`` /
        ``COVARIATE`` columns.
    missing_policy : MissingPolicy
        Policy for ``NaN`` / null handling.

    Returns
    -------
    torch.Tensor
        ``LongTensor`` for categorical encodings, ``FloatTensor``
        otherwise.
    """
    series = df[column]
    dtype = series.dtype
    is_numeric = _is_numeric_dtype(dtype)
    null_count = series.is_null().sum()

    if null_count > 0:
        if missing_policy == MissingPolicy.RAISE:
            raise ValueError(
                f"column {column!r} has {null_count} missing values "
                f"but missing_policy={MissingPolicy.RAISE.value}"
            )
        if missing_policy == MissingPolicy.DROP:
            raise ValueError(
                f"column {column!r}: MissingPolicy.DROP requires the "
                f"caller to pre-filter the dataframe; this function "
                f"encodes the column as given"
            )
        if missing_policy == MissingPolicy.IMPUTE:
            if is_numeric:
                fill = series.mean()
            else:
                # Modal value: take the value with the highest count.
                counts = series.drop_nulls().value_counts(name="_count_")
                fill = counts.sort("_count_", descending=True)[column][0]
            series = series.fill_null(fill)
        # MASK falls through; NaN -> NaN for numeric, -1 code for
        # categorical (handled below).

    if role in (ColumnRole.PLATE_INDEX, ColumnRole.OBJECT):
        if categories is None:
            raise ValueError(
                f"encode_column: role={role.value} requires a "
                f"categories ordering for column {column!r}"
            )
        cat_index = {c: i for i, c in enumerate(categories)}
        values = series.to_list()
        codes = [cat_index[v] if v is not None else -1 for v in values]
        return torch.tensor(codes, dtype=torch.long)

    if role == ColumnRole.OBSERVATION and not is_numeric:
        if categories is None:
            raise ValueError(
                f"encode_column: non-numeric observation column "
                f"{column!r} requires a categories ordering"
            )
        cat_index = {c: i for i, c in enumerate(categories)}
        values = series.to_list()
        codes = [cat_index[v] if v is not None else -1 for v in values]
        return torch.tensor(codes, dtype=torch.long)

    # Numeric observation or covariate path.
    values = series.to_list()
    return torch.tensor(
        [float("nan") if v is None else float(v) for v in values],
        dtype=torch.float32,
    )
