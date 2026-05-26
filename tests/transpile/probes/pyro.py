"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Pyro backend.

Runs the transpiled `def model(y=None): ...` Pyro source through
[`pyro.poutine.trace`][pyro.poutine.trace], substituting observed
values via [`pyro.condition`][pyro.condition] and clamping latent
sample sites with [`pyro.poutine.substitute`][pyro.poutine.substitute].
The trace's `.log_prob_sum()` is the joint log-density.

Available iff `pyro-ppl` (and its `torch` dependency) is importable.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class PyroProbe:
    backend: str = "pyro"

    def available(self) -> bool:
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in ("pyro", "torch")
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
                "pyro / torch not installed; PyroProbe.available() "
                "returned False but evaluate() was called anyway"
            )
        import pyro
        import torch

        namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
        exec(source.decode("utf-8"), namespace)  # noqa: S102
        model = namespace.get("model")
        if not callable(model):
            raise RuntimeError(
                f"pyro probe: transpiled source for {fixture_name!r} "
                f"did not define a `model` callable"
            )

        log_densities: list[float] = []
        for pt in points:
            data_kwargs = {k: _as_tensor(v) for k, v in pt.data.items()}
            param_dict = {k: _as_tensor(v) for k, v in pt.params.items()}
            conditioned = pyro.condition(model, data=param_dict)
            traced = pyro.poutine.trace(conditioned).get_trace(**data_kwargs)
            log_densities.append(float(traced.log_prob_sum()))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"pyro {pyro.__version__}"},
        )


def _as_tensor(value):
    import torch

    if isinstance(value, (int, float)):
        return torch.tensor(float(value))
    return torch.tensor([float(v) for v in value])


_PROBE: LogDensityProbe = PyroProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["PyroProbe"]
