# zed-extension-qvr

A [Zed](https://zed.dev) extension providing syntax highlighting for
Quivers DSL files (`.qvr`). The grammar source lives at
[`grammars/qvr/`](../../grammars/qvr/) in the same repository; this
extension is a thin packaging layer that points Zed at it.

## Install (dev / local)

From a checkout of this repository:

```bash
# Tell Zed where this extension lives.
mkdir -p ~/.config/zed/extensions
ln -s "$(pwd)/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Then in Zed, open the command palette and run **`zed: reload extensions`**.
`.qvr` files now highlight using the tree-sitter grammar at
`grammars/qvr/`.

## Install (when published)

Once this extension is submitted to the public Zed extension registry,
it will be installable from Zed's `extensions:` panel by name (`QVR`).
Until then, the dev / local path above is the supported route.

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
when the grammar's queries change, this copy needs to be refreshed.
A repo-level pre-commit hook keeps the two in sync.
