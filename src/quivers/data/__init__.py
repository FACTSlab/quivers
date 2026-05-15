"""Dataframe-side surface: schema inference, observation packing, and
DSL composition helpers.

Bridges between user dataframes (pandas, polars, or any
[Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible
backend) and the QVR DSL: derives object cardinalities from
``df[col].n_unique()``, builds the per-row plate-index tensors from
deterministic categorical orderings, and emits the ``object``
declarations + ``observations`` dict consumed by inference.

The dataframe library is not a hard dependency. Users install pandas,
polars, or any other Narwhals-supported backend; ``DatasetSchema``
accepts whichever they hand in.
"""

from quivers.data.encoding import (
    ColumnRole,
    MissingPolicy,
    encode_column,
)
from quivers.data.schema import DatasetSchema, compose

__all__ = [
    "ColumnRole",
    "MissingPolicy",
    "DatasetSchema",
    "compose",
    "encode_column",
]
