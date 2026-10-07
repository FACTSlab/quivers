# Chart Semirings

Semiring abstractions for parameterizing chart parsing
([Goodman 1999](https://aclanthology.org/J99-4004/)). Selecting a semiring
changes what the CKY skeleton computes: marginal log-probability, a Viterbi
best parse, boolean recognition, or a derivation count.

::: quivers.stochastic.semiring
    options:
      filters: ["!^_", "!^[A-Z][A-Z_]+$"]

## Semiring singletons

::: quivers.stochastic
    options:
      members:
        - LOG_PROB
        - VITERBI
        - BOOLEAN
        - COUNTING
      show_root_heading: false
      show_root_toc_entry: false
