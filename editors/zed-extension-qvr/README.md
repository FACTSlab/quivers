# zed-extension-qvr

A [Zed](https://zed.dev) extension providing syntax highlighting for
Quivers DSL files (`.qvr`). The grammar source lives at
[`grammars/qvr/`](../../grammars/qvr/) in the same repository; this
extension is a thin packaging layer that points Zed at it.

## Install (dev / local)

In Zed's Extensions view, choose **Install Dev Extension** and select
`editors/zed-extension-qvr` from this checkout. Reinstall the dev
extension after changing its manifest or packaged queries. `.qvr`
files then use the tree-sitter grammar declared by the extension.

## Install (when published)

Once this extension is submitted to the public Zed extension registry,
it will be installable from Zed's `extensions:` panel by name (`QVR`).
Until then, use the dev-extension route above.

## Layout

```
editors/zed-extension-qvr/
├── extension.toml                     extension manifest
├── languages/qvr/
│   ├── config.toml                    Zed language config (file types, comments)
│   └── highlights.scm                 tree-sitter highlight queries
└── README.md
```

The `highlights.scm` is a copy of the canonical
[`grammars/qvr/queries/highlights.scm`](../../grammars/qvr/queries/highlights.scm);
when the grammar's queries change, this copy must be refreshed. The
repository tests validate the canonical query's node kinds, but no
pre-commit hook copies it into the extension.
