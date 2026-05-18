"""Walk every tagged QVR grammar revision and commit a structural
schema to the panproto VCS at ``grammars/qvr/vcs/.panproto/``.

For each git tag whose ``grammars/qvr/grammar.js`` differs from the
previous tag's, this script extracts that tag's ``grammar.json``
(the tree-sitter-generated structural description), translates it
into a :class:`panproto.Schema` keyed by *rule name* (so vertices
named ``composition_decl`` / ``type_decl`` / etc. are shared
across revisions where those rules appear), and commits the result
via the Python :class:`panproto.Repository` API.

Naming vertices by rule keeps the auto-derived migration between
adjacent revisions cheap: panproto's vertex-mapping matches by name,
so unchanged rules are recognised instantly and only the renamed /
new / removed rules need any work. Compare with parsing
``grammar.js`` as a JavaScript AST, which produces anonymous
per-syntax-node vertices that share no labels across revisions and
forces panproto into the combinatorial ``find_best_morphism``
search.

Each VCS commit is tagged with the matching git tag name so a
panproto migration can address any historical surface by the same
identifier git uses.

Usage:

    python grammars/qvr/vcs/build_schemas.py [--reset]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import panproto


_REPO_ROOT = Path(__file__).resolve().parents[3]
_VCS_ROOT = Path(__file__).resolve().parent
_PANPROTO_DIR = _VCS_ROOT / ".panproto"
_GRAMMAR_JS_PATH = "grammars/qvr/grammar.js"
_GRAMMAR_JSON_PATH = "grammars/qvr/src/grammar.json"
_AUTHOR = "qvr-grammar-vcs <vcs@quivers>"
_PROTOCOL_NAME = "qvr-grammar"

# Vertex kind used for every grammar-rule vertex. The custom
# ``qvr-grammar`` protocol declares ``rule`` as its single object
# kind; every vertex (rule names + the synthetic ``source_file``
# root) is committed under this kind.
_RULE_KIND = "rule"


def _qvr_grammar_protocol() -> panproto.Protocol:
    """Return the custom ``qvr-grammar`` protocol used to label
    every grammar-derived Schema."""
    return panproto.define_protocol(
        {
            "name": _PROTOCOL_NAME,
            "schema_theory": "core",
            "instance_theory": "core",
            "edge_rules": [],
            "obj_kinds": [_RULE_KIND],
            "constraint_sorts": [],
        }
    )


def _git_show(spec: str) -> bytes:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "show", spec],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return b""
    return out.stdout


def _ordered_grammar_tags() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "tag", "--sort=v:refname"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [t for t in out.stdout.splitlines() if t.startswith("v")]


def _distinct_grammar_revisions() -> list[tuple[str, bytes]]:
    """Walk tags in semver order; keep the first tag whose
    ``grammar.js`` bytes differ from the previous tag's. Returns
    ``(tag, grammar_json_bytes)`` pairs."""
    seen_hash: str | None = None
    result: list[tuple[str, bytes]] = []
    for tag in _ordered_grammar_tags():
        js = _git_show(f"{tag}:{_GRAMMAR_JS_PATH}")
        if not js:
            continue
        h = hashlib.sha256(js).hexdigest()[:16]
        if h == seen_hash:
            continue
        seen_hash = h
        grammar_json = _git_show(f"{tag}:{_GRAMMAR_JSON_PATH}")
        if not grammar_json:
            continue
        result.append((tag, grammar_json))
    return result


def _current_head_grammar() -> bytes:
    return (_REPO_ROOT / _GRAMMAR_JSON_PATH).read_bytes()


def _rule_field_targets(rule_body: Any) -> set[str]:
    """Recursively walk a tree-sitter grammar.json rule body to
    collect every ``SYMBOL`` it ultimately references.

    The resulting set is the rule's structural fan-out: every
    symbol the rule's RHS can produce, regardless of whether it
    appears under an ``alias``, ``field``, ``seq``, ``choice``,
    ``repeat`` or other combinator.
    """
    targets: set[str] = set()
    stack: list[Any] = [rule_body]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "SYMBOL" and isinstance(
                node.get("name"), str,
            ):
                targets.add(node["name"])
                continue
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return targets


def _build_schema(
    protocol: panproto.Protocol, grammar_json_bytes: bytes,
) -> panproto.Schema:
    """Translate a tree-sitter ``grammar.json`` into a
    :class:`panproto.Schema` of the custom ``qvr-grammar`` protocol.

    Vertices are the grammar's rule names; edges connect each rule
    to every symbol it references in its RHS, labelled by the
    referenced symbol's name (so panproto's auto-derive sees a
    stable edge vocabulary across revisions).
    """
    grammar = json.loads(grammar_json_bytes.decode("utf-8"))
    rules = grammar.get("rules", {})
    if not isinstance(rules, dict):
        raise ValueError("grammar.json: missing or non-dict 'rules' field")

    builder = protocol.schema()

    # Vertex per rule name. Tree-sitter rule names beginning with
    # underscore are "hidden" (inlined into the parent rule); we
    # commit them anyway so the schema graph is precise.
    for rule_name in rules:
        builder.vertex(rule_name, _RULE_KIND)

    # Synthetic root edges aren't required: ``source_file`` is
    # already a rule and connects out via its own RHS references.
    # Each rule's edges follow its structural fan-out.
    for rule_name, body in rules.items():
        for target in sorted(_rule_field_targets(body)):
            if target not in rules:
                # The target is a literal token / string / regex
                # produced by tree-sitter; it has no rule entry, so
                # we skip it rather than emit a dangling edge.
                continue
            builder.edge(rule_name, target, target)

    return builder.build()


def _reset_vcs() -> None:
    if _PANPROTO_DIR.exists():
        shutil.rmtree(_PANPROTO_DIR)
    panproto.Repository.init(str(_VCS_ROOT))


def _commit_and_tag(
    repo: panproto.Repository,
    schema: panproto.Schema,
    message: str,
    tag: str | None,
) -> None:
    repo.add(schema)
    commit_id = repo.commit(message=message, author=_AUTHOR)
    if tag is not None:
        repo.create_tag(tag, commit_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe .panproto/ before rebuilding the schema chain.",
    )
    args = parser.parse_args(argv)

    if args.reset:
        _reset_vcs()
        print("reset panproto VCS")

    protocol = _qvr_grammar_protocol()
    repo = panproto.Repository.open(str(_VCS_ROOT))

    revisions = _distinct_grammar_revisions()
    for tag, grammar_json in revisions:
        schema = _build_schema(protocol, grammar_json)
        _commit_and_tag(
            repo,
            schema,
            message=f"qvr grammar at {tag}",
            tag=tag,
        )
        print(
            f"  {tag}: {schema.vertex_count} rules, "
            f"{schema.edge_count} field edges",
            flush=True,
        )

    head_bytes = _current_head_grammar()
    last_tag_bytes = revisions[-1][1] if revisions else b""
    if head_bytes != last_tag_bytes:
        schema = _build_schema(protocol, head_bytes)
        _commit_and_tag(
            repo,
            schema,
            message="qvr grammar at HEAD: homogenized surface",
            tag=None,
        )
        print(
            f"  HEAD: {schema.vertex_count} rules, "
            f"{schema.edge_count} field edges",
            flush=True,
        )
    else:
        print("  HEAD identical to last tag, no new commit needed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
