# Examples Gallery

Complete `.qvr` programs spanning probabilistic regressions,
latent-variable models, state-space models, language models,
encoder-decoder networks, and weighted deductions. Every example
ships a `## Try it` section with synthetic-data generation, an
SVI fit, and a Bayesian posterior block; all snippets are
executed in CI.

All source files live under `docs/examples/source/`.

## Regressions

- [Bayesian Linear Regression](bayesian-regression.md): Normal likelihood with Normal-prior weights.
- [Beta Regression](beta-regression.md): Beta-distributed response with logit link.
- [Gamma Regression](gamma-regression.md): Gamma response with log link.
- [Horseshoe Regression](horseshoe-regression.md): sparse linear regression under the horseshoe prior.
- [Negative Binomial Regression](negbin-regression.md): overdispersed count response.
- [Zero-Inflated Poisson Regression](zip-regression.md): count response with a separate zero-inflation channel.
- [Item Response Theory (2PL)](irt-2pl.md): Rasch-style logistic IRT with item difficulty and discrimination.
- [Weibull Survival](survival-weibull.md): proportional-hazards survival with Weibull baseline.

## Latent-variable models

- [Factor Analysis](factor-analysis.md): linear factor decomposition with free per-dimension noise.
- [Probabilistic PCA](ppca.md): isotropic-noise special case of factor analysis.
- [Latent Dirichlet Allocation](lda.md): topic model with Dirichlet priors on per-document and per-topic distributions.
- [Gaussian Mixture Model](mixture-model.md): finite mixture with grouped marginalization over the cluster label.
- [Variational Autoencoder](vae.md): amortized inference over a continuous latent with neural decoder.
- [Bayesian Neural Network](bnn.md): nonlinear regression through an MLP-parameterised Normal kernel, with Normal priors lifted over every weight.
- [Parametric Partial Pooling](parametric-pooling.md): random effects from a parametric program template, with a labeled return tuple, a score-step sum-to-zero factor, and export selection.
- [Probabilistic Matrix Factorization](pmf.md): low-rank Bayesian completion of a sparse rating matrix.
- [Bilinear Tensor Contraction](tensor-contraction.md): neural-tensor-layer scoring of predicate-argument pairs via an operadic three-way contraction.

## State-space and time-series models

- [AR(1)](ar1.md): first-order autoregressive Normal scalar series.
- [Changepoint Model](changepoint.md): piecewise-constant means with a sampled change point.
- [Stochastic Volatility](stochastic-volatility.md): AR(1) log-variance driving Normal returns.
- [Linear-Gaussian SSM](linear-gaussian-ssm.md): Kalman-filter dynamics with learnable transition / emission.
- [Hidden Markov Model (Discrete)](hmm.md): row-stochastic transition and emission matrices over finite states.
- [Hidden Markov Model (Continuous)](continuous-hmm.md): Gaussian-emission HMM with continuous-state latent trajectory.
- [Deep Markov Model](deep-markov.md): nonlinear-MLP transition and emission cells.

## Language models

- [Vanilla RNN LM](vanilla-rnn-lm.md): single-cell Elman RNN scored by a Categorical head.
- [GRU LM](gru-lm.md): gated recurrent unit cell.
- [LSTM LM](lstm-lm.md): long short-term memory cell.
- [Bidirectional RNN LM](bidirectional-rnn-lm.md): forward + backward RNN with a masked-token target.
- [Transformer LM](transformer-lm.md): stacked self-attention with a Categorical head.

## Encoder-decoder and structured outputs

- [Sequence-to-Sequence](seq2seq.md): encoder-decoder with attention scored by a target-side Categorical.
- [Tree-Structured Categorical](tree-categorical.md): hierarchical Categorical observation over a parametric tree.

## Weighted deductions

- [PCFG](pcfg.md): probabilistic context-free grammar with learnable production weights.
- [CCG](ccg.md): combinatory categorial grammar with forward / backward application and composition.
- [Type-Logical Grammar (Lambek)](type-logical.md): Lambek calculus with residuated slashes and tensor.
- [PMCFG](pmcfg.md): probabilistic multiple context-free grammar with WH-movement via a rank-2 non-terminal.
- [Schema-Bundled Chart Parser](schema-chart-parser.md): pattern-polymorphic rule schemas bundled into a differentiable CKY chart parser.
- [Term Autoencoder](term-autoencoder.md): a signature, encoder, decoder, and loss compressing typed lambda terms.
- [Multimodal TLG](multimodal-tlg.md): Lambek calculus extended with diamond and box modalities.
- [Custom Sequent Rules](custom-rules.md): user-defined sequents over a free residuated category.
- [Quantifier Scope](quantifier-scope.md): continuation-monad lift for generalized quantifiers.
- [Montague NLI](montague-nli.md): Montague-style lambda-term LFs plus modus-ponens NLI prover.
