# Interactive surface: REPL, kernel, language server

quivers ships an interactive type-explorer and a matching language
server. Both share the same parser
([panproto](https://github.com/panproto/panproto) tree-sitter), the
same elaborator
([`quivers.dsl.Compiler`](../api/dsl/compiler.md)), and the same highlight
table, so the colour you see in the TUI matches the colour your
editor shows over LSP, which matches what gets emitted as Jupyter
output. One source of truth, four surfaces.

This page is the comprehensive reference. For a one-paragraph
overview see [Quickstart](../getting-started/quickstart.md).

## Installation

quivers requires **Python 3.14 or newer**. The Textual TUI and the
language server are optional extras; the base install gives you the
library plus the `qvr check` subcommand.

### With `pip`

```bash
pip install 'quivers[repl,lsp]'
```

### With `uv` as a global tool

```bash
uv tool install --python 3.14 'quivers[repl,lsp]'
```

`uv tool install` puts the `qvr`, `qvr-lsp`, and `qvr-kernel`
console scripts on your PATH in a managed virtual environment so
the install does not collide with any project venv.

### From a local checkout (editable)

```bash
uv tool install --reinstall --python 3.14 --editable \
    /path/to/quivers --with 'quivers[repl,lsp]'
```

The `--editable` flag points the install at the working tree so
local changes take effect immediately. The trailing `--with`
re-adds the optional extras (uv strips them otherwise).

### What the extras bring in

| Extra | Pulls in |
| --- | --- |
| `[repl]` | [Textual](https://textual.textualize.io/), [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/), [rich](https://rich.readthedocs.io/), [ipykernel](https://ipykernel.readthedocs.io/) |
| `[lsp]` | [pygls](https://github.com/openlawlibrary/pygls), `lsprotocol` |

Optional renderers that ``:plate --open`` will pick up if they are
on PATH but are not installed automatically:

| Optional dependency | What it does for the REPL |
| --- | --- |
| [`daft`](https://docs.daft-pgm.org/) (via `pip install daft`) | Renders ``:plate PROGRAM --open`` to a publication-quality PNG via matplotlib |
| [Graphviz](https://graphviz.org/) (the ``dot`` binary) | Renders ``:plate`` / ``:graph`` ``--open`` to PNG via ``dot -Tpng`` |
| LaTeX with ``tikz-network`` / ``bayesnet`` | Compiles the ``--tikz`` output |

If neither `daft` nor `dot` is available, `:plate --open` falls
back to printing Mermaid source with a note suggesting
mermaid.live.

### Verifying the install

```bash
qvr --version
qvr check docs/examples/source/lda.qvr      # batch-validate one or more files
qvr repl --help                              # confirm the TUI is reachable
```

### After install

| Command | What it does |
| --- | --- |
| `qvr repl` | Textual TUI (or prompt_toolkit fallback if stdin is not a TTY) |
| `qvr repl --plain` | Force the single-line prompt_toolkit frontend |
| `qvr repl FILE.qvr` | Load FILE on startup, then drop to the prompt |
| `qvr lsp` / `qvr-lsp` | Run the Language Server over stdio (use this in editor config) |
| `qvr kernel install --user` / `qvr-kernel install --user` | Register a Jupyter kernelspec named `quivers` |
| `qvr check FILE...` | Batch parse + compile; non-zero exit code on any error |

## The REPL

### Starting it

```bash
qvr repl                                              # blank session
qvr repl docs/examples/source/seq2seq.qvr             # load on startup
qvr repl --plain                                      # single-line prompt_toolkit mode
qvr repl --plain < commands.txt                       # non-TTY scripted mode
```

When stdin is a TTY and Textual is importable, the four-pane TUI
launches. `--plain` forces the prompt_toolkit single-line frontend.

### Layout (TUI)

```
+---- Status bar -----------------------------------------+
|                                                         |
|  Input editor                  |  Env filter            |
|  (multi-line TextArea          +-----------------------+
|   with bracket pairing,        |                       |
|   auto-indent, Tab completion) |  Env tree             |
|                                |  (objects/spaces/     |
+--------------------------------+   morphisms/rules,    |
|  Output log                    |   click any leaf to   |
|  (history of evaluated         |   :info it)           |
|   commands and rendered        |                       |
|   responses)                   |                       |
+--------------------------------+-----------------------+
|  Watches  (hidden when empty; pinned :watch results)    |
+---------------------------------------------------------+
|  Diagnostics (hidden when empty; errors with locations) |
+---------------------------------------------------------+
|  Footer: visible key bindings                           |
```

The watches and diagnostics strips collapse when there is nothing to
show, so unused panels never cost screen real estate.

### Status bar

The top bar shows, separated by `·`:

- the loaded file path (or `<no file>`),
- the active algebra (`ProductFuzzyAlgebra`, `LogProbAlgebra`, …),
- counts: `N obj · M space · K morph · J rule`.

It refreshes after every evaluation, reload, or watch update.

### Environment browser

The right-hand `Tree` widget groups bindings by namespace:

- `objects`, every `object` and `alias` declaration
- `spaces`, every `space` and `type` declaration
- `morphisms`, `latent`, `observed`, `kernel`, `embed`, `program`, `let`
- `rules`, `rule` declarations

The root node auto-expands on every refresh; each namespace expands
its leaves. Click a leaf (or arrow-key to it and press Enter) to fire
`:info NAME` in the output log.

Above the tree is a one-line **filter input**. Type any substring
and the tree fuzzy-collapses to the bindings whose names contain it.
Clear the filter to bring everything back.

### Output log

Rendered responses appear here in source order. Bodies tagged as QVR
(every `:type`, `:info`, `:doc`, `:dump`, `:watch`, `:browse` result)
are passed through the shared
[`tokenize`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py)
pipeline and rendered with the
[`STYLE_TABLE`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py)
colours. Identifiers known to the env are upgraded to their semantic
colour even when the surrounding line wasn't a parseable QVR
declaration; the same identifier always reads the same way.

Identifiers rendered as type / function / namespace are also
**clickable links**. Clicking any name fires `:info NAME`. The
`-- declared at PATH:LINE:COL` footer underneath each `:info` body
is also clickable; it opens that file at that line in `$EDITOR`
(falling back to `$VISUAL`, then `vi`).

### Input editor

The input pane is a multi-line `TextArea` with three quality-of-life
behaviours layered on:

1. **Bracket pairing.** Typing `(`, `[`, or `{` inserts the matching
   closer and parks the cursor between them.
2. **Auto-indent on Enter.** Lines that end with `(`, `[`, `{`, `:`,
   `->`, `=`, or `<-` get one extra indent on the next line so
   `program` / `deduction` blocks stay aligned without effort.
3. **Tab completion** (described below).

Press the eval key (next section) to evaluate the buffer; the input
is cleared on success.

### Meta-commands

GHCi-shaped, leading `:`. The dispatcher is exact-match: use one of
the full names below or the explicit short alias in the second
column.

| Command | Short | What it does |
| --- | --- | --- |
| `:load FILE` | `:l` | Parse + elaborate, rebind the session env |
| `:reload` | `:r` | Re-`:load` the last file, print added/removed/changed names |
| `:type EXPR` | `:t` | Print EXPR's resolved type as canonical QVR (`morphism f : A -> B [role=latent]`, `object X : FinSet 3`, `object Z : Real 64`) |
| `:kind T` | `:k` | Print T's AST variant and enumerate the sibling TypeExpr variants |
| `:info NAME` | `:i` | Show NAME's declaration verbatim from the source, plus its location and doc comment. Pass `--python` for the didactic AST `repr()` instead |
| `:doc NAME` |  | Render only the doc comment(s) for NAME |
| `:browse [PATH]` | `:b` | List every binding, optionally filtered by a top-level namespace (`objects`/`spaces`/`morphisms`/`rules`/...) or by a `::`-separated scope path (`lda`, `lda::z`, `CCG::fwd_app`). Paths show that binding's inner scope. |
| `:plate PROGRAM` | `:p` | Render PROGRAM's plate diagram. Default is a Rich table; flags `--mermaid` / `--dot` / `--tikz` / `--daft` emit alternative formats; `--open` renders to PNG via daft or graphviz and opens with the system viewer. |
| `:graph PROGRAM` | `:g` | Vertical step-flow view of PROGRAM (one row per sample / observe / let / marginalize / return step with its dependency parents). Same `--mermaid` / `--dot` / `--open` flags as `:plate`. |
| `:where NAME` |  | List every scope path whose final segment is NAME (top-level and nested). Useful for finding every site a binding is referenced. |
| `:effects PROGRAM` |  | Compare PROGRAM's declared `[effects=[...]]` set against the effect set the compiler infers from the body. |
| `:shape PROGRAM` |  | Pretty-print PROGRAM's `ChainShape`: per-step depth, kind, intermediate axis size, source location. |
| `:dump NAME [--json]` |  | Pretty-print NAME's AST node (`--json` for didactic's `model_dump_json`) |
| `:edit NAME` |  | Open `$EDITOR` on NAME's source, splice the edited text back into the module, recompile |
| `:trace EXPR` |  | Step through elaboration of a morphism expression, surfacing each intermediate domain/codomain |
| `:save [FILE]` | `:s` | Write the live module to FILE (or back to the loaded path) via [`module_to_source`](../api/dsl/emit.md) |
| `:watch EXPR` | `:w` | Pin EXPR for re-eval on every recompile; result appears in the Watches strip |
| `:unwatch [EXPR]` |  | Remove one watch, or clear all when no argument is given |
| `:set k=v` |  | Toggle session options (`highlight`, `unicode`, `show_axes`, `paranoid`, `autoload_on_save`, `theme`) |
| `:help [CMD]` | `:h` | Without arg: full command list. With one: detailed help for CMD |
| `:quit` | `:q` / `:exit` | Exit the REPL |

A bare line (no leading `:`) is evaluated as **either** appended
statements (parsed, compiled into the live module, env updated)
**or**, if parsing as statements fails, treated as an expression and
piped through `:type`.

### Command reference

Every meta-command in full, with concrete usage examples on the
[LDA gallery example](../examples/lda.md). Load it first:

```
qvr repl docs/examples/source/lda.qvr
```

#### `:load FILE` / `:l FILE`

Parse and elaborate `FILE`, rebinding the session environment to
the result. Replaces any previously loaded module. Diagnostics
(parse errors, constraint violations, compile errors) populate
the bottom strip and move the cursor selection to the first error
location.

```
> :load docs/examples/source/hmm.qvr
loaded hmm.qvr (3 obj  1 prog)
> :l docs/examples/source/lda.qvr
loaded lda.qvr (3 obj  1 prog)
```

#### `:reload` / `:r`

Re-execute `:load` against the file that was loaded most recently,
then print the diff (added / removed / changed bindings). Triggers
automatically every second when the source file's mtime advances,
so saving in `$EDITOR` updates the live env without leaving the
REPL.

```
> :reload
reloaded lda.qvr
  + morphism foo : Word -> Word
  - sample tau
```

#### `:type EXPR` / `:t EXPR`

Print `EXPR`'s resolved type as a single QVR-shaped line. Accepts
either a bare identifier, a `::` scope path, or a general
expression that the parser can elaborate.

```
> :type lda
program lda(alpha : Real, beta : Real) : Word -> Word

> :type lda::theta
sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]

> :type lda::z::w
observe w : Word <- Categorical(phi[z]) [via=word_idx]

> :type Doc
object Doc : FinSet 20
```

#### `:kind T` / `:k T`

Print `T`'s AST variant class plus the sibling variants of the
same tagged union. Useful for understanding the AST shape under
the surface syntax.

```
> :kind FinSet 20
DiscreteConstructor
  siblings: TypeName, ObjectProduct, ObjectCoproduct,
            ObjectSlash, ObjectEffectApply,
            ContinuousConstructor
```

#### `:info NAME [--python]` / `:i NAME`

Show NAME's declaration verbatim from the source (with the
doc comment if any), plus the source location. With `--python`,
prints the didactic AST `repr()` instead.

```
> :info lda
program lda(alpha : Real, beta : Real) : Word -> Word
    sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(beta) [over=Word, iid_over=Topic]
    marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
        observe w : Word <- Categorical(phi[z]) [via=word_idx]
    return theta
-- declared at docs/examples/source/lda.qvr:32:0

> :info lda::theta
sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]
-- sample-site inside lda at docs/examples/source/lda.qvr:33:4
```

#### `:doc NAME`

Render only the doc comment for NAME (lines prefixed with `#!` in
the source). Returns "(no doc comment)" when none exists. Works
with scoped paths as well.

```
> :doc lda
(no doc comment)
```

#### `:browse [PATH]` / `:b [PATH]`

Without an argument, lists every binding in the module grouped by
top-level kind (objects / spaces / morphisms / rules / programs /
deductions / signatures / encoders / decoders / losses / bundles
/ contractions). With a `::`-path argument, lists the binding's
inner scope.

```
> :browse
objects:
  Doc : FinSet 20
  Topic : FinSet 3
  Word : FinSet 200
programs:
  lda(alpha : Real, beta : Real) : Word -> Word

> :browse lda
program lda(alpha : Real, beta : Real) : Word -> Word
  param alpha : Real
  param beta : Real
  sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]
  sample phi : Topic <- Dirichlet(beta) [over=Word, iid_over=Topic]
  marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
  return theta

> :browse lda::z
marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
  observe w : Word <- Categorical(phi[z]) [via=word_idx]
```

#### `:plate PROGRAM [--mermaid|--dot|--tikz|--daft|--open]` / `:p PROGRAM`

Render PROGRAM's plate-notation diagram. Default is a Rich table
listing every variable with its kind, family, plate stack, and
dependency parents. Flags emit alternate sources or open a
rendered image.

```
> :plate lda
plate graph of lda  (Word -> Word)

#  kind            variable  family       plates      parents
-  --------------  --------  -----------  ----------  -------
1  ○ latent        theta     Dirichlet    Doc         alpha
2  ○ latent        phi       Dirichlet    Topic       beta
3  ⊘ marginalized  z         Categorical  Topic       theta
4  ● observed      w         Categorical  Doc x Word  phi, z

plates:
  Doc [20]
  Topic [3]
  Word [200] parent=Doc
```

| Flag | Output |
| --- | --- |
| (default) | Rich table in the TUI; plain ASCII in `--plain` mode |
| `--mermaid` | Mermaid `graph TD` source. Paste into mermaid.live or a markdown file |
| `--dot` | Graphviz DOT source. Pipe through `dot -Tpng > plate.png` |
| `--tikz` | LaTeX TikZ + `tikz-network` snippet. Drop into a LaTeX document |
| `--daft` | Python script using the [`daft`](https://docs.daft-pgm.org/) library. Run it to produce SVG / PNG / PDF |
| `--open` | Render to a temp PNG via daft or `dot`, open with the system default viewer. Falls back to printing Mermaid source if no renderer is installed |

Badges in the table view:

| Badge | Kind |
| --- | --- |
| `○` | latent (unobserved sample) |
| `●` | observed (clamped at runtime) |
| `⊘` | marginalized (integrated out by an enclosing marginalize block) |
| `·` | deterministic (a `let` binding) |

#### `:graph PROGRAM [--mermaid|--dot|--open]` / `:g PROGRAM`

Vertical step-flow view of PROGRAM. One row per program step
(sample / observe / let / marginalize / return) with its
dependency parents on the side. Differs from `:plate` in that
the rows are *steps* (program-source order) rather than
*variables* (which a `let` can introduce mid-stream).

```
> :graph lda
program lda: Word -> Word

#  kind          step                                           parents
-  ------------  ---------------------------------------------  -------
1  latent        sample theta : Doc <- Dirichlet(alpha)         alpha
2  latent        sample phi : Topic <- Dirichlet(beta)          beta
3  marginalized  marginalize z : Topic <- Categorical(theta)    theta
4  observed      observe w : Doc x Word <- Categorical(phi, z)  phi, z
```

Same `--mermaid` / `--dot` / `--open` flags as `:plate`.

#### `:where NAME`

List every scope path in the loaded module whose final segment
is `NAME`, sorted by depth (top-level first). Useful for finding
every site a binding is referenced.

```
> :where theta
references to 'theta':
  sample-site    lda::theta

> :where Doc
references to 'Doc':
  object         Doc
```

#### `:effects PROGRAM`

Compare PROGRAM's declared `[effects=[...]]` option block against
the effect set the compiler infers from the body's step kinds
(`Sample` from sample-sites, `Score` from observes, `Marginal`
from marginalize blocks, `Pure` otherwise).

```
> :effects lda
program lda:
  declared : {(none)}
  inferred : {Marginal, Sample, Score}
  (no [effects=[...]] declared)
```

If the program declares `[effects=[Pure]]` but the body contains
a sample site, the output flags a `! leak` line. If it declares
`[effects=[Sample, Score]]` but the body never observes, the
output flags an `! unused` line.

#### `:shape PROGRAM`

Pretty-print PROGRAM's `ChainShape`: per-step depth, kind,
intermediate axis size, source location. Useful for auditing
chain depth before fitting.

```
> :shape lda
chain shape (log_prob):
  #  depth  kind         name        size
  -  -----  -----------  ----------  ----
  1   1      latent       theta       20
  2   2      latent       phi         3
  3   3      marginalize  z           3
  4   4      observe      w           200
```

#### `:dump NAME [--json]`

Print NAME's AST node. Default is Python `repr()`; `--json` emits
the didactic `model_dump_json(indent=2)` so the structure is
machine-readable.

```
> :dump lda::theta
SampleStep(vars=('theta',), morphism='Dirichlet', args=('alpha',), …)
```

#### `:edit NAME`

Open `$EDITOR` on NAME's source slice, then splice the edited
text back into the module and recompile. Saves the resulting
source via [`module_to_source`](../api/dsl/emit.md). When the
edit doesn't parse, the original declaration is retained and the
parse error appears in the diagnostics strip.

```
> :edit lda
   (vim opens on the lda program; on quit, the new lda compiles
   and the env is rebound)
```

#### `:trace EXPR`

Step through elaboration of `EXPR` as a morphism expression. For
each intermediate sub-expression, surfaces the inferred
domain / codomain. Useful for understanding why a composition
fails to type-check.

```
> :trace f >> g
f : A -> B
g : B -> C
(f >> g) : A -> C
```

#### `:save [FILE]` / `:s [FILE]`

Write the live module to `FILE` (or back to the loaded path if
no argument) via
[`module_to_source`](../api/dsl/emit.md). Captures any
`:edit`-driven changes and any statements typed at the prompt.

```
> :save my_edits.qvr
saved my_edits.qvr
```

#### `:watch EXPR` / `:w EXPR`

Pin `EXPR` to the Watches strip. The pinned expression is
re-evaluated after every recompile, so each `:reload` (or every
edit on disk under the file watcher) updates the displayed
value. The strip is hidden when no watches are pinned.

```
> :watch lda::theta
> :watch sigmoid(0.5)
```

#### `:unwatch [EXPR]`

Remove one watch (when `EXPR` matches an existing one) or all
watches (when called with no argument).

```
> :unwatch lda::theta
> :unwatch
```

#### `:set KEY=VALUE`

Toggle session options. The options table:

| Key | Default | What it does |
| --- | --- | --- |
| `highlight` | `true` | Apply syntax highlighting to output |
| `unicode` | `true` | Use Unicode glyphs (×, →, ⊗, ⊘) in pretty-prints |
| `show_axes` | `true` | When printing morphisms, annotate each axis with its name |
| `paranoid` | `false` | Re-run constraint checks after every recompile |
| `autoload_on_save` | `true` | Re-load the file when its mtime advances |
| `theme` | `ansi_dark` | Rich/Pygments syntax theme; see the [Themes](#themes) section below for the full list |

```
> :set show_axes=true
> :set paranoid=true
```

#### `:help [CMD]` / `:h`

In the TUI, opens the modal help dialog (also bound to `F1`) with
the full command list grouped by category and a live filter
input. In `--plain` mode, prints the help text to stdout. With an
argument, prints detailed help for one command.

```
> :help plate
:plate PROGRAM [--mermaid|--dot|--tikz|--daft|--open]
  Render PROGRAM's plate-notation diagram. …
```

#### `:quit` / `:q` / `:exit`

Exit the REPL. The Textual binding `Ctrl-Q` does the same.

### Scope paths (`::`)

Every meta-command that takes a binding name accepts a
`::`-separated scope path. The leftmost segment is a top-level
declaration; each subsequent segment looks up a named child in
that declaration's scope. Examples on the LDA gallery example:

```
> :type lda::theta
sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]

> :type lda::z::w
observe w : Word <- Categorical(phi[z]) [via=word_idx]

> :browse lda::z
marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
  observe w : Word <- Categorical(phi[z]) [via=word_idx]

> :where theta
references to 'theta':
  sample-site    lda::theta
```

What each container kind exposes as named children:

| Container | Children |
|---|---|
| `program` | typed parameters + body steps (sample / observe / let / marginalize / score / return) |
| `marginalize` | the same set of program-step kinds, nested inside the marginalize block |
| `deduction` | rules + atoms + lexicon entries |
| `signature` | sorts + constructors + binders + vertex_kinds + edge_kinds |
| `encoder` (inline body) | op_rules + init_rules + message_rules + update_rules + var_inits |
| `decoder` | per-constructor heads |
| `bundle` | member rule names |
| `contraction` | declared input parameter names |
| `object` / `space` / `morphism` / `rule` / `loss` / `category` | none (leaf) |

The env tree on the right side of the TUI carries each binding's
full `::` path on its node data; clicking any node, top-level or
nested, fires `:info PATH` against that exact binding. Tab
completion completes `lda::` to the program's children, `lda::z::`
to the inner marginalize's children, and so on. A bare-name
prefix like `thet` also surfaces scoped descendants (`lda::theta`)
so users discover nested bindings without typing the prefix.

### Program exploration

Five commands work together for understanding a program's
structure:

* `:graph PROGRAM`, vertical step-flow view (one row per program
  step with its dependency parents). The TUI default; pair with
  `--mermaid` / `--dot` / `--open` for external formats.
* `:plate PROGRAM`, plate-notation diagram of the program's
  random variables. Default is a Rich table with one row per
  variable + the plates it inhabits + its dependency parents. Flags
  `--mermaid`, `--dot`, `--tikz`, `--daft` emit alternate sources;
  `--open` renders to PNG via daft or graphviz and launches the
  system viewer.
* `:where NAME`, every scope path that references NAME (including
  cross-references between programs and rules).
* `:effects PROGRAM`, declared `[effects=[...]]` set vs the set
  the compiler infers from the body. Catches `[effects=[Pure]]`
  programs that accidentally contain a sample.
* `:shape PROGRAM`, per-step `ChainShape`: step kind, chain
  depth, intermediate axis size, source location. Useful for
  auditing chain depth before fitting.

### Key bindings

Selected for cross-platform reliability, every binding here reaches
the application on macOS, Linux, and Windows without per-terminal
configuration.

| Key | Action |
| --- | --- |
| `Ctrl-G` | Evaluate the buffer |
| `Ctrl-O` | Evaluate the buffer (alternate) |
| `F8` | Evaluate the buffer (Fn-row alternate) |
| `Ctrl-Up` | Recall the previous input from history |
| `Ctrl-Down` | Advance to the next input |
| `Tab` | Cycle completion candidates at the cursor |
| `Ctrl-P` | Open the command palette (fuzzy meta-command picker) |
| `Ctrl-L` | Clear the eval log |
| `Ctrl-R` | Reload the loaded file |
| `Ctrl-Q` | Quit |
| `F1` | Show the meta-command help in the eval log |
| `Ctrl-Enter`, `Ctrl-J` | Evaluate (hidden fallback; only fires in terminals that forward Ctrl-Enter, like Wezterm/Kitty/Windows Terminal/iTerm2 with a CSI-u keymap) |

Inside the TextArea, all of Textual's emacs-style editing bindings
remain available: `Ctrl-A`/`Ctrl-E` line nav, `Ctrl-W` delete word
back, `Ctrl-K` kill to end of line, `Ctrl-X`/`Ctrl-C`/`Ctrl-V`
cut/copy/paste, `Ctrl-Z`/`Ctrl-Y` undo/redo. The eval bindings
above use `priority=True` so they fire even though `Ctrl-G` is not
otherwise a TextArea action.

> **macOS Fn-row note.** `F5` is reserved by macOS Dictation, `F3`
> and `F4` by Mission Control and Launchpad, `F11` by Show Desktop.
> `F8` is the only function key macOS doesn't already claim by
> default. Enable "Use F1, F2, etc. keys as standard function keys"
> in System Settings → Keyboard if you want to use F-keys without
> holding Fn.

> **Ctrl-Enter, Alt-Enter, Cmd-Enter.** These never reach the
> application on macOS Terminal.app or the default iTerm2 profile.
> The terminal emulator drops them at the wire layer. To make them
> work you have to configure your emulator to send them; for iTerm2
> map them to `\x1b[13;5u` (the CSI-u "modifyOtherKeys" encoding),
> for Windows Terminal use the same string. Wezterm, Kitty, and
> Alacritty forward them out of the box.

### Tab completion

`Tab` at the cursor:

1. Builds a candidate list from four sources:
   - Meta-command names (`:load`, `:type`, …) when the prefix starts with `:`.
   - Env names: every object, space, morphism, rule in the live env, tagged with its namespace.
   - QVR grammar keywords (`latent`, `observed`, `program`, `over`, `iid`, `via`, …) and builtins (`Normal`, `Euclidean`, `softmax`, …).
   - File-system paths after `:load`.
2. Inserts the first candidate, replacing the prefix under the cursor.
3. Subsequent `Tab` presses cycle through the remaining candidates without rebuilding the list.

The completer is the exact one the LSP and the Jupyter kernel call;
the surfaces never disagree about what's available.

### Command palette

`Ctrl-P` opens Textual's command palette pre-populated with every
meta-command. Type any substring to fuzzy-filter; Enter selects.
Selecting a meta-command inserts it into the input pane with a
trailing space, ready for the argument. This is the discoverable
substitute for memorising shortcuts.

### Input history

Every evaluated line is appended to `~/.config/quivers/history`
(`$XDG_CONFIG_HOME/quivers/history` if set). The file is plain
newline-separated entries; the same history is loaded the next time
you start `qvr repl`.

- `Ctrl-Up` walks backwards from the most recent entry.
- `Ctrl-Down` walks forward; at the end you return to an empty
  buffer.
- The current input is replaced wholesale; the TextArea is not
  modal, so just keep typing to continue editing.

The prompt_toolkit fallback (`qvr repl --plain`) uses the same file
through `prompt_toolkit.history.FileHistory`.

### File watcher / auto-reload

When a file is loaded, the TUI polls its mtime once per second. If
the mtime advances (your editor saved it), the session re-runs
`:reload` automatically and logs `auto-reload` in the eval log. Any
diagnostics from the new parse show up immediately in the
Diagnostics strip; any pinned `:watch` expressions re-evaluate
against the new env.

The option is on by default and controlled by
`:set autoload_on_save=true|false`.

### Watches

`:watch EXPR` pins EXPR for re-evaluation after every recompile ,
every `:load`, `:reload`, bare-line statement, or autoreload. The
result appears in a dedicated **Watches** strip (auto-hidden when
empty) above the diagnostics strip. Format: `watch EXPR => result`,
syntax-highlighted.

```
qvr> :watch f
watch f => morphism f : Alpha -> Beta [role=latent]
qvr> object Gamma : FinSet 7
qvr> morphism g : Alpha -> Gamma [role=latent]
qvr> :watch g
# both f and g are now pinned; modifying Alpha or Beta updates both
```

`:unwatch EXPR` removes one; `:unwatch` alone clears every watch.

### Inline diagnostic markers

Every error diagnostic with a known location (parse errors, compile
errors, constraint violations) selects the offending span in the
input pane so you see where the failure originated. The Diagnostics
strip shows the same message in human-readable form:

```
[error] compile:5:12: undefined object 'X'
```

The strip is hidden when there are no diagnostics.

### Click handlers

Two surfaces are click-aware:

- **Identifiers in rendered output.** Clicking a type-, function-,
  or namespace-coloured identifier fires `:info NAME` in the eval
  log. Works on any text the renderer painted with one of those
  semantic colours: env-known names, grammar-classified types,
  built-in functions, algebras.
- **`PATH:LINE:COL` footers.** The `-- declared at ...` line under
  every `:info` body is a clickable link that launches
  `$EDITOR +LINE PATH` (e.g. `vi +5 model.qvr`,
  `code --goto model.qvr:5`).

### Themes

`:set theme=NAME` controls the Rich syntax theme used when bodies
are rendered through `rich.syntax.Syntax`. Useful values:

- `ansi_dark` (default), `ansi_light`
- `monokai`, `dracula`, `solarized-dark`, `solarized-light`
- `nord`, `gruvbox-dark`, `github-dark`
- any other [Pygments style](https://pygments.org/styles/) shipped with your install

The Textual app itself follows your terminal's colour scheme; the
theme option only affects fenced QVR / JSON blocks in `:info` and
`:dump` output.

### Bracket pairing and auto-indent

In the input TextArea:

- `(`, `[`, `{` → insert the matching closer, leave cursor between.
- Enter after `(`, `[`, `{`, `:`, `->`, `=`, `<-` → add one indent
  level (four spaces by default) on the new line.
- Indentation is otherwise preserved verbatim.

These rules are heuristic; they cover the common shape of QVR
`program` and `deduction` blocks without needing a full grammar
configuration.

### Plain mode

`qvr repl --plain` (or any non-TTY invocation) uses
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/)
instead of Textual. The same `ReplSession` drives the prompt, so
every meta-command works identically; the only differences are:

- Single-line input, with prompt_toolkit's built-in history
  navigation (Up/Down) instead of `Ctrl-Up`/`Ctrl-Down`.
- Output is rendered through [Rich](https://rich.readthedocs.io/) if
  available, plain text otherwise.
- No env tree, no watches panel, no command palette, no click
  handlers.

This is the mode CI uses for scripted runs: pipe a plain text file
of meta-commands and bare statements (one per line) on stdin.

## The Jupyter kernel

`qvr-kernel install --user` registers a `quivers` kernelspec. After
that, every Jupyter frontend (Notebook, JupyterLab, VS Code
notebooks, `jupyter console`) sees QVR as a first-class language.

```bash
qvr-kernel install --user
jupyter console --kernel quivers
```

```
In [1]: :load model.qvr
loaded model.qvr: 17 binding(s)

In [2]: :type backbone
morphism backbone : Source * Target -> Combined [role=latent]

In [3]: object Extra : FinSet 8
        morphism g : Extra -> Combined [role=latent]
installed module: 19 binding(s)
```

Cell semantics:

- Leading-`:` lines are dispatched as meta-commands.
- Other lines are appended to the live module and compiled.
- Blank lines separate independent chunks within a cell, so you can
  mix meta-commands and declarations freely.

Notebook-side features:

- **Tab completion** routes through the same completer as the TUI
  and LSP.
- **Inspect (`Shift-Tab`)** routes through `:info`, showing the
  declaration in the inspector panel.
- The kernelspec advertises `pygments_lexer: qvr` so cell content
  is highlighted by the bundled Pygments lexer.

## The Language Server

`qvr-lsp` speaks LSP 3.17 over stdio. Pointed at any LSP-aware
editor it provides hover, go-to-definition, references, document
symbols, semantic highlighting, completion, formatting, and live
diagnostics.

### VS Code / Cursor

The [`vscode-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr)
extension in the repository ships:

- the TextMate grammar (for the initial render before the LSP
  attaches),
- a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node)
  bridge that auto-discovers `qvr-lsp` in:
  1. an explicit `qvr.lsp.path` setting (supports `${workspaceFolder}` expansion),
  2. `<workspace>/.venv/bin/qvr-lsp` (uv / venv convention),
  3. `<workspace>/.venv/Scripts/qvr-lsp.exe` (Windows venv),
  4. `$VIRTUAL_ENV/bin/qvr-lsp`,
  5. plain `qvr-lsp` on `$PATH`.

Settings:

| Key | Default | Purpose |
| --- | --- | --- |
| `qvr.lsp.enabled` | `true` | Master toggle |
| `qvr.lsp.path` | `qvr-lsp` | Override the executable path |
| `qvr.lsp.args` | `[]` | Extra CLI arguments |

Install the packaged `.vsix` directly:

```bash
cd editors/vscode-qvr
npm install
npx tsc -p .
npx @vscode/vsce package --allow-missing-repository --no-yarn
code --install-extension vscode-qvr-*.vsix         # or cursor --install-extension
```

### Zed

The [`zed-extension-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr)
extension exports `qvr-lsp` as a `language_server`. Symlink the
extension into Zed's extensions directory:

```bash
mkdir -p ~/.config/zed/extensions
ln -s "$PWD/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Reload extensions from the command palette. Zed spawns `qvr-lsp` on
`$PATH` automatically when you open a `.qvr` file.

### Neovim

```lua
require("lspconfig").configs.qvr = {
  default_config = {
    cmd = { "qvr-lsp" },
    filetypes = { "qvr" },
    root_dir = require("lspconfig.util").root_pattern(
      "pyproject.toml", ".git"
    ),
  },
}
require("lspconfig").qvr.setup({})
```

### Capabilities

Every capability advertised by `qvr-lsp`:

| LSP method | What it returns |
| --- | --- |
| `textDocument/publishDiagnostics` | Parser, constraint-solver, and compiler diagnostics with source ranges |
| `textDocument/hover` | The declaration as a fenced `qvr` block, with the didactic AST `repr()` stacked beneath under a collapsed `<details>` so both forms are always available |
| `textDocument/definition` | Jump to the originating declaration |
| `textDocument/references` | Every textual occurrence of the name |
| `textDocument/documentSymbol` | All top-level declarations grouped by symbol kind (`Class` for objects/spaces, `Function` for morphisms, `Variable` otherwise) |
| `textDocument/completion` | Env names + grammar keywords + builtins + paths, same source as the REPL completer |
| `textDocument/semanticTokens/full` | Env-aware semantic token stream driven by the shared [`STYLE_TABLE`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py) |
| `textDocument/formatting` | Canonical re-emission via [`module_to_source`](../api/dsl/emit.md) (no-op when the module contains a statement variant the canonical emitter doesn't yet cover) |
| `textDocument/didOpen` / `didChange` / `didSave` / `didClose` | Incremental sync, full re-analysis per change |

### Hover format

A hover always shows two visually-separated panes:

````markdown
Optional doc comment lines if the decl had `#!` doc-comment lines above it.

**QVR source**

<!-- compile: false -->
```qvr
morphism f : Alpha -> Beta [role=latent]
```

---

**AST (didactic)**

<details><summary><i>click to expand</i></summary>

<!-- python: skip -->
```python
MorphismDecl(morphism_kind='latent', name='f', domain=TypeName(...), ...)
```

</details>
````

- The QVR block is sliced from the original source so user formatting
  and comments survive verbatim.
- A horizontal rule draws the divider between the panes.
- The Python AST is the didactic `repr()`, hidden by default in the
  collapsed `<details>` element so it only takes vertical space when
  you click "click to expand".

## Architecture

All four surfaces fan out from one class:

```
                    ┌─────────────────────┐
                    │   ReplSession       │  pure Python state +
                    │  (state + dispatch) │  meta-command dispatch
                    └────────┬────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
   ┌────▼─────┐         ┌────▼─────┐          ┌────▼─────┐
   │ Textual  │         │ prompt_  │          │ Jupyter  │
   │ TUI      │         │ toolkit  │          │ kernel   │
   └──────────┘         └──────────┘          └──────────┘

                       ┌────────────────────┐
                       │  qvr-lsp           │  builds its own
                       │  (pygls)           │  per-document state
                       └────────────────────┘  but shares the
                                               highlighter +
                                               completer
```

Components shared across all surfaces:

- [`quivers.cli.repl_session`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_session.py), `ReplSession`, the meta-command dispatcher, the live env, the watch list.
- [`quivers.cli.repl_complete`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_complete.py), `all_completions(session, prefix)`; fans out to env, grammar, paths.
- [`quivers.cli.repl_highlight`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py), `tokenize`, `STYLE_TABLE`, `to_rich_text`, `to_semantic_token_data`; one classifier feeds every renderer.

Each frontend is a thin adapter:

- [`quivers.cli.repl_tui`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_tui.py). Textual app.
- [`quivers.cli.repl_prompt`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_prompt.py), prompt_toolkit single-line frontend.
- [`quivers.kernel.quivers_kernel`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/kernel/quivers_kernel.py), ipykernel adapter.
- [`quivers.lsp.server`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/lsp/server.py), pygls server.

This is the seam that keeps the four surfaces in sync: anywhere a
user sees `Source`, it's classified by the same call and rendered
with the same colour, regardless of which frontend is asking.
