"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Edward2 backend.

Edward2 random variables compose into a `tape()` context: each
`ed.Distribution(...)` constructor call inside the tape records the
RV; after the tape closes, the joint log-density is the sum of
`rv.distribution.log_prob(rv.value)` over every captured RV.

The transpiled source from `qvr-edward2` is a plain `def model(...): ...`
whose body constructs the RVs via positional `tfd.<Family>(args,
name="...")` calls. The probe execs the bytes, calls the model
inside an `ed.tape()`, then substitutes per-RV values from the
test point and sums the per-RV log-densities.

Available iff `tensorflow_probability.edward2` is importable.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class Edward2Probe:
    backend: str = "edward2"

    def available(self) -> bool:
        return importlib.util.find_spec("tensorflow_probability") is not None

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        if not self.available():
            raise RuntimeError(
                "tensorflow_probability not installed; "
                "Edward2Probe.available() returned False but "
                "evaluate() was called anyway"
            )
        import tensorflow as tf
        from tensorflow_probability import edward2 as ed

        namespace: dict[str, object] = {
            "edward2": ed,
            "ed": ed,
            "tf": tf,
        }
        exec(source.decode("utf-8"), namespace)  # noqa: S102
        model = namespace.get("model")
        if not callable(model):
            raise RuntimeError(
                f"edward2 probe: transpiled source for {fixture_name!r} "
                f"did not define a `model` callable"
            )

        log_densities: list[float] = []
        for pt in points:
            data_kwargs = {k: _as_tensor(v) for k, v in pt.data.items()}
            value_map = {**pt.params, **pt.data}
            # Tape the model under value-set interception: each RV
            # call inside the tape uses the value from `value_map`
            # when the RV's name matches a key.
            with ed.interception(_make_value_setter(value_map)):
                with ed.tape() as recorded:
                    model(**data_kwargs)
            total = sum(
                rv.distribution.log_prob(rv.value)
                for rv in recorded.values()
            )
            log_densities.append(float(tf.reduce_sum(total).numpy()))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "tensorflow_probability.edward2"},
        )


def _make_value_setter(value_map):
    """Build an Edward2 interceptor that clamps RVs by name."""
    import tensorflow as tf

    def interceptor(rv_constructor, *args, **kwargs):
        name = kwargs.get("name")
        if name is not None and name in value_map:
            kwargs["value"] = tf.constant(value_map[name])
        return rv_constructor(*args, **kwargs)
    return interceptor


def _as_tensor(value):
    import tensorflow as tf

    return tf.constant(value)


_PROBE: LogDensityProbe = Edward2Probe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["Edward2Probe"]
