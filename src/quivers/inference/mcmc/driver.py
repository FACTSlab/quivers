"""MCMC orchestrator: chains, warmup, and posterior diagnostics.

The :class:`MCMC` driver wraps a :class:`MCMCKernel` and applies
it to a model: it builds the registry, draws an initial position
(from the prior, an existing guide, or a user-supplied dict),
runs ``num_warmup`` adaptation steps with
``kernel.start_adaptation()``, freezes the kernel via
``kernel.stop_adaptation()``, and then collects ``num_samples``
post-warmup samples per chain.

Chains are run sequentially within a process. For parallel chains
across a real multi-core / multi-GPU workload, wrap an
:class:`MCMC` instance in a :class:`torch.multiprocessing` pool —
the driver is stateless across runs.

The result is an :class:`MCMCResult` carrying per-site posterior
draws (already pushed through the constraint bijectors so they
sit in the model's natural support) plus split-:math:`\\hat R`
(Vehtari et al. 2021, `doi:10.1214/20-BA1221
<https://doi.org/10.1214/20-BA1221>`_) and effective sample size
diagnostics computed across chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
)
from quivers.inference.registry import LatentRegistry


InitStrategy = Literal["prior", "zero", "guide"]


@dataclass
class MCMCResult:
    """Posterior samples and per-chain diagnostics.

    Attributes
    ----------
    samples : dict[str, torch.Tensor]
        Per-site posterior draws on the constrained support.
        Shape ``(num_chains, num_samples, *site_shape)``.
    log_densities : torch.Tensor
        Unconstrained-space log-density (Jacobian-corrected) at
        every posterior draw. Shape ``(num_chains, num_samples)``.
    acceptance_rates : torch.Tensor
        Per-chain post-warmup acceptance rate. Shape
        ``(num_chains,)``.
    divergence_counts : torch.Tensor
        Per-chain post-warmup divergence count. Shape
        ``(num_chains,)``.
    r_hat : dict[str, torch.Tensor]
        Per-site split-:math:`\\hat R`. Each site's tensor has the
        site's shape (one scalar per coordinate).
    ess : dict[str, torch.Tensor]
        Per-site effective sample size. Same shape convention as
        :attr:`r_hat`.
    num_warmup : int
    num_samples : int
    """

    samples: dict[str, torch.Tensor]
    log_densities: torch.Tensor
    acceptance_rates: torch.Tensor
    divergence_counts: torch.Tensor
    r_hat: dict[str, torch.Tensor]
    ess: dict[str, torch.Tensor]
    num_warmup: int
    num_samples: int

    @property
    def num_chains(self) -> int:
        return int(self.log_densities.shape[0])

    @property
    def mean_acceptance(self) -> float:
        return float(self.acceptance_rates.mean().item())

    @property
    def total_divergences(self) -> int:
        return int(self.divergence_counts.sum().item())


def _split_rhat_1d(chains: torch.Tensor) -> torch.Tensor:
    """Split-:math:`\\hat R` along axis 0 (chains) × axis 1
    (samples) of a 2-axis tensor. Returns a scalar.

    Implementation follows Vehtari et al. 2021 equation 3; we split
    each chain in half so the within-chain variance estimator picks
    up trend pathologies.
    """
    n_chains, n_samples = chains.shape
    half = n_samples // 2
    if half < 2:
        return torch.tensor(float("nan"))
    split_chains = torch.cat([chains[:, :half], chains[:, half : 2 * half]], dim=0)
    m = float(split_chains.shape[0])
    n = float(split_chains.shape[1])
    chain_means = split_chains.mean(dim=1)
    grand_mean = chain_means.mean()
    B = n * ((chain_means - grand_mean) ** 2).sum() / (m - 1)
    chain_vars = split_chains.var(dim=1, unbiased=True)
    W = chain_vars.mean()
    var_hat = ((n - 1) / n) * W + B / n
    if W.item() <= 0:
        return torch.tensor(float("nan"))
    return torch.sqrt(var_hat / W)


def _ess_1d(chains: torch.Tensor) -> torch.Tensor:
    """Effective sample size for one coordinate across chains.

    Uses the standard initial monotone sequence estimator
    (Geyer 1992) with the Vehtari et al. 2021 correction for
    cross-chain pooling.
    """
    n_chains, n_samples = chains.shape
    if n_samples < 4:
        return torch.tensor(float("nan"))
    centered = chains - chains.mean(dim=1, keepdim=True)
    # Compute autocorrelations via FFT, averaged over chains.
    pad = 1
    while pad < 2 * n_samples:
        pad *= 2
    f = torch.fft.rfft(centered, n=pad, dim=1)
    acov = torch.fft.irfft(f * torch.conj(f), n=pad, dim=1)[:, :n_samples]
    acov = acov / torch.arange(n_samples, 0, -1, dtype=acov.dtype)
    var_plus = acov.mean(dim=0)
    if var_plus[0].item() <= 0:
        return torch.tensor(float("nan"))
    rho = var_plus / var_plus[0]
    # Sum pairs until the sum-of-pairs goes non-positive.
    sum_rho = torch.tensor(-1.0)
    t = 0
    while t < n_samples - 1:
        pair = rho[t] + rho[t + 1] if t + 1 < n_samples else rho[t]
        if pair.item() <= 0:
            break
        sum_rho = sum_rho + 2 * pair
        t += 2
    return torch.tensor(float(n_chains * n_samples) / max(1.0, float(sum_rho)))


def _diagnostics(
    samples: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute per-site :math:`\\hat R` and ESS over the
    ``(num_chains, num_samples, *site_shape)`` posterior tensors."""
    r_hat: dict[str, torch.Tensor] = {}
    ess: dict[str, torch.Tensor] = {}
    for name, draws in samples.items():
        n_chains = draws.shape[0]
        n_samples = draws.shape[1]
        flat = draws.reshape(n_chains, n_samples, -1)
        n_coords = flat.shape[-1]
        rh = torch.empty(n_coords)
        es = torch.empty(n_coords)
        for c in range(n_coords):
            rh[c] = _split_rhat_1d(flat[:, :, c])
            es[c] = _ess_1d(flat[:, :, c])
        site_shape = tuple(draws.shape[2:]) or (1,)
        r_hat[name] = rh.reshape(site_shape)
        ess[name] = es.reshape(site_shape)
    return r_hat, ess


class MCMC:
    """MCMC chain runner.

    Parameters
    ----------
    kernel : MCMCKernel
        Markov kernel (e.g. :class:`HMCKernel`, :class:`NUTSKernel`).
    num_warmup : int
        Number of adaptation steps. The kernel's adaptation
        machinery (dual averaging, Welford covariance) runs over
        this prefix.
    num_samples : int
        Post-warmup samples per chain.
    num_chains : int
        Independent chains. Default ``4`` (Stan / NumPyro default).
    init_strategy : {"prior", "zero", "guide"}
        How to pick each chain's initial position.
    """

    def __init__(
        self,
        kernel: MCMCKernel,
        num_warmup: int,
        num_samples: int,
        num_chains: int = 4,
        init_strategy: InitStrategy = "prior",
    ) -> None:
        if num_warmup < 0:
            raise ValueError(f"MCMC: num_warmup must be >= 0, got {num_warmup}")
        if num_samples < 1:
            raise ValueError(f"MCMC: num_samples must be >= 1, got {num_samples}")
        if num_chains < 1:
            raise ValueError(f"MCMC: num_chains must be >= 1, got {num_chains}")
        self.kernel = kernel
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.init_strategy = init_strategy

    def _initial_position(
        self,
        registry: LatentRegistry,
        guide: Guide | None,
        chain_idx: int,
    ) -> torch.Tensor:
        D = registry.total_unconstrained_dim
        if self.init_strategy == "zero":
            return torch.zeros(D)
        if self.init_strategy == "guide":
            if guide is None:
                raise ValueError(
                    "MCMC.run: init_strategy='guide' requires a guide argument"
                )
            # Sample the guide once, pull out the unconstrained
            # values, flatten.
            x = torch.zeros(1, 1)
            constrained = guide.rsample(x)
            unc: dict[str, torch.Tensor] = {}
            for site in registry.sites.values():
                v = constrained[site.name]
                if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 1):
                    v_e = v.unsqueeze(-1)
                else:
                    v_e = v
                if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
                    v_e = v_e[0]
                z = site.bijector.inv(v_e)
                unc[site.name] = z
            return registry.flatten_unconstrained(unc)
        # "prior": draw a fresh standard-normal position, scaled
        # mildly so chains spread out without diverging.
        torch.manual_seed(int(chain_idx))
        return 0.1 * torch.randn(D)

    def _to_constrained(
        self,
        registry: LatentRegistry,
        z_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        per_site_unc = registry.unflatten_unconstrained(z_flat)
        out: dict[str, torch.Tensor] = {}
        for site in registry.sites.values():
            z_site = per_site_unc[site.name]
            if not site.is_plate:
                z_site = z_site.unsqueeze(0)
            v = site.bijector(z_site)
            if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                v = v.squeeze(-1)
            out[site.name] = v
        return out

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        """Run the configured kernel for ``num_chains`` chains of
        ``num_warmup + num_samples`` steps each."""
        observed_names = set(observations.keys())
        registry = LatentRegistry.from_model(model, observed_names)
        potential = PotentialFn(model, registry, x, observations)

        site_shapes: dict[str, tuple[int, ...]] = {
            site.name: site.constrained_shape or (1,)
            for site in registry.sites.values()
        }
        per_chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_shapes}
        per_chain_log_density = torch.empty(self.num_chains, self.num_samples)
        per_chain_accept = torch.empty(self.num_chains)
        per_chain_divergences = torch.empty(self.num_chains)

        for chain in range(self.num_chains):
            init_pos = self._initial_position(registry, guide, chain)
            state = self.kernel.init(registry, model, x, observations, init_pos)
            divergence_count = 0
            # Warmup.
            if self.num_warmup > 0:
                self.kernel.start_adaptation()
                for _ in range(self.num_warmup):
                    state = self.kernel.step(state, potential)
                    if state.diverged:
                        divergence_count += 1
                self.kernel.stop_adaptation()
            # Reset accept-count so the reported rate excludes warmup.
            sampling_state = KernelState(
                position=state.position,
                log_density=state.log_density,
                grad_log_density=state.grad_log_density,
                step_count=state.step_count,
                accept_count=0,
                diverged=False,
                extras=state.extras,
            )
            chain_samples: dict[str, list[torch.Tensor]] = {n: [] for n in site_shapes}
            sampling_divergences = 0
            sampling_accept = 0
            for s in range(self.num_samples):
                sampling_state = self.kernel.step(sampling_state, potential)
                if sampling_state.diverged:
                    sampling_divergences += 1
                if sampling_state.extras.get("accept_prob", 0.0) > 0:
                    sampling_accept += 1
                draws = self._to_constrained(registry, sampling_state.position)
                for n, v in draws.items():
                    chain_samples[n].append(v)
                per_chain_log_density[chain, s] = sampling_state.log_density
            per_chain_accept[chain] = sampling_state.accept_count / float(
                self.num_samples
            )
            per_chain_divergences[chain] = float(sampling_divergences)
            for n, draws_list in chain_samples.items():
                stacked = torch.stack(draws_list, dim=0)
                per_chain_samples[n].append(stacked)

        # Stack per-chain → (num_chains, num_samples, *site_shape).
        samples: dict[str, torch.Tensor] = {}
        for n, chain_draws in per_chain_samples.items():
            samples[n] = torch.stack(chain_draws, dim=0)

        r_hat, ess = _diagnostics(samples)
        return MCMCResult(
            samples=samples,
            log_densities=per_chain_log_density,
            acceptance_rates=per_chain_accept,
            divergence_counts=per_chain_divergences,
            r_hat=r_hat,
            ess=ess,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
        )


__all__ = ["MCMC", "MCMCResult", "InitStrategy"]
