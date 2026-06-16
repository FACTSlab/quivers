"""In-container Edward2 probe.

Uses the standalone ``edward2`` package (``google/edward2``); the
TFP-bundled ``tensorflow_probability.edward2`` submodule was retired.
The trace API is ``ed.tape`` and the value-override API is
``ed.condition`` (replacement for the old ``ed.interception``).

The probe computes joint log-density by tracing the model once
(under any conditioning whose value-shape matches the
``RandomVariable``'s ``value.shape``) and then evaluating
``rv.distribution.log_prob(target_value)`` against the user-supplied
value tensor. Observed variables whose target value shape exceeds
the RV's ``value.shape`` (the canonical case: a ``Normal(loc=mu)``
where ``mu`` is itself a vector-shaped deterministic input expression)
are paired with the distribution and scored directly: the model's
RV value is not used, so its sample shape is irrelevant.
"""
import json
import pathlib

import tensorflow as tf
import edward2 as ed


def _tensor(value):
    """Cast a JSON-decoded payload into a TF tensor.

    Defaults float values to ``tf.float32`` to match TFP's default
    distribution dtype; lists of ints become ``tf.int32`` (matching
    ``Bernoulli`` / ``Categorical`` integer support). Mixing dtypes
    across the model graph (e.g. a ``float32`` sampled RV multiplied
    by a ``float64`` design-matrix input) raises in TF's strict
    type-checked ops.
    """
    if isinstance(value, list):
        if value and all(isinstance(v, bool) for v in value):
            return tf.constant(value, dtype=tf.int32)
        if value and all(isinstance(v, int) for v in value):
            return tf.constant(value, dtype=tf.int32)
        return tf.constant(value, dtype=tf.float32)
    if isinstance(value, bool):
        return tf.constant(value, dtype=tf.int32)
    if isinstance(value, int):
        return tf.constant(value, dtype=tf.int32)
    if isinstance(value, float):
        return tf.constant(value, dtype=tf.float32)
    return tf.constant(value)


def _cast_for_rv(rv, raw):
    """Cast ``raw`` to a tensor matching the random variable's dtype."""
    return tf.constant(raw, dtype=rv.distribution.dtype)


def _matches_value_shape(rv, raw_tensor) -> bool:
    """True iff ``raw_tensor.shape`` matches ``rv.value.shape``.

    The Edward2 ``condition`` tracer overwrites the RV's value with
    the conditioned tensor before constructing the RV, but the
    ``RandomVariable`` constructor validates the value's shape against
    ``sample_shape + batch_shape + event_shape``. A mismatch raises;
    we conservatively detect it here so the second trace can either
    condition only the matching RVs (and score the rest separately)
    or bail out of conditioning entirely for that name.
    """
    rv_shape = tuple(int(d) for d in rv.value.shape)
    raw_shape = tuple(int(d) for d in raw_tensor.shape)
    return rv_shape == raw_shape


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())

    ns = {"edward2": ed, "ed": ed, "tf": tf}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    for pt in points:
        params = pt.get("params", {})
        data = pt.get("data", {})
        data_kw = {k: _tensor(v) for k, v in data.items()}

        # Pass 1: trace the model with no value overrides. This gives
        # the per-name `RandomVariable` so we can read each
        # distribution's dtype and `value.shape`.
        with ed.tape() as probe_tape:
            model(**data_kw)

        target_values: dict[str, tf.Tensor] = {}
        for name, raw in {**params, **data}.items():
            if name not in probe_tape:
                continue
            target_values[name] = _cast_for_rv(probe_tape[name], raw)

        # Conditioning a name whose target value shape does not match
        # the RV's `value.shape` would raise in the RV constructor;
        # restrict the second trace's overrides to the safe set, and
        # score the rest by direct `dist.log_prob(value)` evaluation
        # against the first-trace distribution.
        safe_overrides: dict[str, tf.Tensor] = {}
        direct_scores: dict[str, tf.Tensor] = {}
        for name, target in target_values.items():
            if _matches_value_shape(probe_tape[name], target):
                safe_overrides[name] = target
            else:
                direct_scores[name] = target

        with ed.condition(**safe_overrides):
            with ed.tape() as recorded:
                model(**data_kw)

        log_terms = []
        for name, rv in recorded.items():
            if name in direct_scores:
                # Use the conditioned distribution and the user-supplied
                # value (which carries the true observation shape).
                log_terms.append(
                    tf.reduce_sum(rv.distribution.log_prob(direct_scores[name]))
                )
            else:
                log_terms.append(
                    tf.reduce_sum(rv.distribution.log_prob(rv.value))
                )
        total = tf.add_n(log_terms)
        log_densities.append(float(total.numpy()))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
