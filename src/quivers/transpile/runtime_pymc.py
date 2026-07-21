"""Runtime helpers grafted into emitted PyMC programs.

PyMC ships no continuous-Bernoulli distribution, so the renderer
grafts the [`ContinuousBernoulli`][quivers.transpile.runtime_pymc.ContinuousBernoulli]
wrapper below into the emitted module. The wrapper builds a
[`pymc.CustomDist`][pymc.CustomDist] whose log-density is the exact
continuous-Bernoulli density (including its parameter-dependent
normalising constant), so the emitted program's joint log-density
matches the QVR model's `torch.distributions.ContinuousBernoulli`
term.

The renderer parses this file through panproto's Python tree-sitter
grammar at module-load time and copies each top-level
`function_definition` subtree into the per-render schema, so the
emitted source carries a real function definition rather than a
string of source or an `exec`.
"""

from __future__ import annotations

import numpy as np
import pymc
import pytensor.tensor as pt


def _continuous_bernoulli_logp(value, lam):
    """Log-density of the continuous-Bernoulli distribution with
    probability parameter ``lam`` on the unit interval.

    The density is ``C(lam) * lam**x * (1 - lam)**(1 - x)`` with
    normaliser ``C(lam) = 2 * arctanh(1 - 2 * lam) / (1 - 2 * lam)``,
    which has a removable singularity at ``lam == 0.5`` where the
    constant equals ``2``."""
    z = 1.0 - 2.0 * lam
    log_norm = pt.switch(
        pt.lt(pt.abs(z), 1e-6),
        pt.log(2.0),
        pt.log(2.0 * pt.arctanh(z) / z),
    )
    return (
        value * pt.log(lam)
        + (1.0 - value) * pt.log(1.0 - lam)
        + log_norm
    )


def _continuous_bernoulli_random(lam, rng=None, size=None):
    """Inverse-CDF sampler for the continuous-Bernoulli distribution.

    With ``a = log(lam / (1 - lam))`` the CDF inverts to
    ``x = log(1 + (2 * lam - 1) * u / (1 - lam)) / a`` for a uniform
    draw ``u``; at ``lam == 0.5`` the distribution is uniform on the
    unit interval, so ``x == u``."""
    lam = np.asarray(lam, dtype="float64")
    draw_size = size if size is not None else lam.shape
    u = rng.uniform(size=draw_size)
    near_half = np.abs(2.0 * lam - 1.0) < 1e-6
    safe_lam = np.where(near_half, 0.25, lam)
    a = np.log(safe_lam / (1.0 - safe_lam))
    shifted = np.log(1.0 + (2.0 * safe_lam - 1.0) * u / (1.0 - safe_lam)) / a
    return np.where(near_half, u, shifted)


def ContinuousBernoulli(name, probs, **kwargs):
    """Continuous-Bernoulli random variable for PyMC.

    Wraps [`pymc.CustomDist`][pymc.CustomDist] with the exact
    continuous-Bernoulli log-density, so the family is available under
    the same call shape as a built-in ``pymc.<Family>(name, ...)``
    constructor."""
    return pymc.CustomDist(
        name,
        probs,
        logp=_continuous_bernoulli_logp,
        random=_continuous_bernoulli_random,
        dtype="float64",
        **kwargs,
    )
