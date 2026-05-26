"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the NumPyro backend.

Runs the transpiled `def model(y=None): ...` source through
[`numpyro.infer.util.log_density`][numpyro.infer.util.log_density]
at every test point. The probe is in-process: it `exec`s the
transpiled bytes in a fresh namespace, fishes out the ``model``
callable, then evaluates the joint log-density at each ``(θ, y)``.

Available iff `numpyro` (and its `jax` dependency) is importable.
The probe gates on that at construction time; if either is missing,
`available()` returns False and the test layer skips the cell.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class NumPyroProbe:
    backend: str = "numpyro"

    def available(self) -> bool:
        """True iff `numpyro` is importable in-process."""
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in ("numpyro", "jax")
        )

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
                "numpyro / jax not installed; NumPyroProbe.available() "
                "returned False but evaluate() was called anyway"
            )
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer.util import log_density

        namespace: dict[str, object] = {
            "numpyro": numpyro,
            "jnp": jnp,
        }
        # ``exec`` is the right primitive here: the transpiled source
        # is a single ``def model(...): ...``; we need the function
        # object back in the host namespace so we can call it. The
        # transpiled bytes never come from user input, so the
        # template-injection concern that normally guards against
        # ``exec`` does not apply.
        exec(source.decode("utf-8"), namespace)  # noqa: S102
        model = namespace.get("model")
        if not callable(model):
            raise RuntimeError(
                f"numpyro probe: transpiled source for {fixture_name!r} "
                f"did not define a `model` callable"
            )

        log_densities: list[float] = []
        for pt in points:
            # ``log_density`` takes (model, model_args, model_kwargs,
            # params). Observed data is passed as model_args /
            # model_kwargs; latent params are passed as the params
            # dict. The transpiled NumPyro source has
            # ``def model(<observed>=None)``, so each observed var
            # appears as a kwarg.
            data_kwargs = {k: _as_array(v) for k, v in pt.data.items()}
            param_dict = {k: _as_array(v) for k, v in pt.params.items()}
            lp, _ = log_density(model, (), data_kwargs, param_dict)
            log_densities.append(float(lp))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"numpyro {numpyro.__version__}"},
        )


def _as_array(value):
    """Convert a Python scalar / list to a JAX array."""
    import jax.numpy as jnp

    if isinstance(value, (int, float)):
        return jnp.array(value)
    return jnp.array(value)


_PROBE: LogDensityProbe = NumPyroProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["NumPyroProbe"]
