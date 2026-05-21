# MCMC

Markov-chain Monte Carlo: the [`HMC`](#quivers.inference.mcmc.HMCKernel)
and [`NUTS`](#quivers.inference.mcmc.NUTSKernel) kernels, the
[`MCMC`](#quivers.inference.mcmc.MCMC) runner, and the
[`MCMCResult`](#quivers.inference.mcmc.MCMCResult) summary.

The runner targets a `MonadicProgram` directly. For models that
declare every latent as a `sample` site this is immediate; for
models with `nn.Parameter`s or intermediate latent sites the
[`bayesian_lift_parameters`](lifts.md#quivers.inference.lifts.bayesian_lift_parameters)
lift produces the matching `MonadicProgram`.

::: quivers.inference.mcmc
