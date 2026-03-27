# Tutorials

The tutorials section provides guided walkthroughs of quivers' core capabilities. Each tutorial builds concrete examples from scratch, progressing from basic morphisms through stochastic relations, continuous probabilistic programs, linguistic modeling, and variational inference.

## Prerequisites

- Python >= 3.12, PyTorch, and quivers installed (see [Installation](../getting-started/installation.md))
- Familiarity with basic category theory (morphisms, composition, functors)
- Knowledge of quantales and $\mathcal{V}$-enrichment (see [Core Types & Quantales](../guides/core.md))

## Tutorials

### [1. Your First Quiver](first-quiver.md)
Create a simple $\mathcal{V}$-enriched category from scratch. Define FinSet objects, construct both observed and latent morphisms, compose them with the `>>` operator, and inspect the resulting tensors. Covers the basic workflow: what a quiver is in this context (a directed graph with lattice-valued edges), and how to work with morphisms as differentiable PyTorch tensors.

### [2. Stochastic Relations](stochastic-relations.md)
Build models in the FinStoch category: create Markov kernels (StochasticMorphisms), compose them, and apply Bayesian operations like conditioning. Learn how discretized continuous distributions (DiscretizedNormal, DiscretizedBeta) allow embedding continuous randomness into finite-set relations. Demonstrates prob, marginal_prob, and expectation operations.

### [3. Probabilistic Programs](probabilistic-programs.md)
Construct MonadicPrograms by hand using the Python API: define continuous spaces, create conditional distribution families (ConditionalNormal, ConditionalBernoulli), build a program with draw and let steps, sample from it, and compute log-densities. Covers destructuring and the distinction between observed and latent sites.

### [4. Fuzzy Logic Factorization](fuzzy-factorization.md)
Factorize an observed fuzzy relation into a composition of learnable morphisms under the product fuzzy logic quantale. Covers observed and latent morphisms, noisy-OR composition, BCE loss training, and alternative quantales.

### [5. Variational Inference](variational-inference.md)
Set up and run inference on a probabilistic program. Create a MonadicProgram, condition it on observed data, construct an AutoNormalGuide, set up ELBO and SVI, run a training loop, and use Predictive for posterior sampling. Demonstrates the full workflow from model definition to fitted inference.

## How to Use These Tutorials

Read them in order. Each tutorial:

1. States what you will learn
2. Walks through a complete worked example with code
3. Explains key concepts as they arise
4. Points to API documentation for deeper dives

Run all code in an interactive Python environment. Modify examples to build intuition.
