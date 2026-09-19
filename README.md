<h1 align="center">Quivers</h1>

<p align="center">
  <em>A typed functional probabilistic programming language with a PyTorch runtime.</em>
</p>

<p align="center">
  <a href="https://github.com/FACTSlab/quivers/actions/workflows/ci.yml"><img src="https://github.com/FACTSlab/quivers/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://FACTSlab.github.io/quivers"><img src="https://github.com/FACTSlab/quivers/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/quivers/"><img src="https://img.shields.io/pypi/v/quivers" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14%2B-blue" alt="Python 3.14+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://FACTSlab.github.io/quivers/tutorials/qvr/01-first-model/"><strong>Tutorial</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/examples/"><strong>Examples</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/reference/qvr/"><strong>Language reference</strong></a>
  ·
  <a href="https://FACTSlab.github.io/quivers/api/"><strong>Python API</strong></a>
</p>

---

Quivers is a typed probabilistic programming language for compositional
models. A QVR source file is parsed, checked, and lowered to one intermediate
representation used by the PyTorch runtime, editor tooling, and transpilers.
The Python API provides variational and Monte Carlo inference, while the
compiler emits models for eleven probabilistic programming systems.

## A model in QVR

This stochastic GRU language-model step draws update and reset gates, updates
the hidden state, and scores the observed next token.

<!-- GitHub Linguist fallback: QVR currently uses the R lexer. -->
```R
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128

morphism gate_z, gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
morphism lm_head : Hidden -> Token ~ Categorical

program gru_step(x_t, h_prev) : Embedded * Hidden -> Token
    sample z <- gate_z(x_t, h_prev)
    sample r <- gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev
    sample h_candidate <- Normal(reset_hidden, 0.5)

    let h = (1.0 - z) * h_prev + z * h_candidate
    observe next_token <- lm_head(h)
    return next_token

export gru_step
```

Install Quivers, download the full recurrent model, and check it through the
command line:

```bash
pip install quivers
curl -LO https://raw.githubusercontent.com/FACTSlab/quivers/main/docs/examples/source/gru_lm.qvr
qvr check gru_lm.qvr
```

The [tutorial](https://FACTSlab.github.io/quivers/tutorials/qvr/01-first-model/)
starts with the core language and inference workflow. The
[examples gallery](https://FACTSlab.github.io/quivers/examples/) includes
hierarchical and state-space models, mixture models, neural language models,
formal grammars, and structural autoencoders.

## Documentation

- [QVR language reference](https://FACTSlab.github.io/quivers/reference/qvr/)
- [Tutorials](https://FACTSlab.github.io/quivers/tutorials/)
- [Examples](https://FACTSlab.github.io/quivers/examples/)
- [Guides](https://FACTSlab.github.io/quivers/guides/)
- [Python API](https://FACTSlab.github.io/quivers/api/)
- [Semantics](https://FACTSlab.github.io/quivers/semantics/)

## Installation and editors

Quivers requires Python 3.14 or later. Install optional interactive and editor
support with:

```bash
pip install 'quivers[repl,lsp]'
```

The repository includes extensions for
[VS Code and Cursor](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr)
and [Zed](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr).
Both provide immediate syntax highlighting; `qvr-lsp` adds typed diagnostics,
hover, definitions, references, completion, formatting, and semantic tokens.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are welcome at
[github.com/FACTSlab/quivers](https://github.com/FACTSlab/quivers).

## Acknowledgments

This project was developed by [Aaron Steven White](https://aaronstevenwhite.io/)
at the University of Rochester with support from the National Science
Foundation (NSF-BCS-2237175 *CAREER: Logical Form Induction*,
NSF-BCS-2040831 *Computational Modeling of the Internal Structure of Events*).
It was architected and implemented with the assistance of Claude Code.

## License

MIT. See [LICENSE](LICENSE).
