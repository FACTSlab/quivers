# MCMC

Markov-chain Monte Carlo: the [`HMC`](#quivers.inference.mcmc.HMCKernel)
and [`NUTS`](#quivers.inference.mcmc.NUTSKernel) kernels, the
[`MCMC`](#quivers.inference.mcmc.MCMC) runner, and the
[`MCMCResult`](#quivers.inference.mcmc.MCMCResult) summary.

The runner targets a `MonadicProgram`. Models that declare every latent
as a `sample` site need no conversion. For models with `nn.Parameter`s
or intermediate latent sites, the
[`bayesian_lift_parameters`](lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
lift returns a 3-tuple `(MonadicProgram, torch.Tensor, dict[str, torch.Tensor])`,
whose first element is the matching `MonadicProgram`. Callers pass
that first element to the runner.

::: quivers.inference.mcmc
