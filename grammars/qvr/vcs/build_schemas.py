"""Walk every tagged QVR grammar revision and commit it to the
panproto VCS at ``grammars/qvr/vcs/.panproto/``.

The grammar's tree-sitter source is itself a JavaScript file that
panproto's vendored ``javascript`` protocol understands. For each
git tag whose ``grammars/qvr/grammar.js`` differs from the previous
tag's, we extract that file via ``git show``, stage it with
``schema add`` (which parses it through panproto's JS grammar into
a content-addressed schema), commit the result to the VCS, and tag
the VCS commit with the matching git version.

Once every distinct revision is in the VCS plus the current HEAD,
panproto's migration engine can lower any historical ``.qvr``
source to the head grammar via the schema chain. No hand-written
representative ``.qvr`` programs are needed: the schema *is* the
grammar.

Usage:

    python grammars/qvr/vcs/build_schemas.py [--reset]

``--reset`` rebuilds the VCS from scratch (wipes ``.panproto/``
first). Without it, the script is a no-op when every distinct
tag is already committed; pass the flag explicitly to recompute
from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_VCS_ROOT = Path(__file__).resolve().parent
_GRAMMAR_PATH = "grammars/qvr/grammar.js"
_WORK_DIR = Path("/tmp/qvr-vcs-build")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def _tag_grammar(tag: str) -> bytes | None:
    """Return the grammar.js bytes at ``tag``, or ``None`` if absent."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "show", f"{tag}:{_GRAMMAR_PATH}"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None
    if not out.stdout:
        return None
    return out.stdout


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _ordered_grammar_tags() -> list[str]:
    """Return git tags in semver order. Skips lightweight tags by
    relying on ``git tag --sort=v:refname``."""
    out = _run(
        ["git", "-C", str(_REPO_ROOT), "tag", "--sort=v:refname"],
    )
    return [t for t in out.stdout.splitlines() if t.startswith("v")]


def _distinct_grammar_revisions() -> list[tuple[str, bytes]]:
    """Walk tags in order, keep the first tag where the grammar
    bytes changed from the previous tag's grammar."""
    seen_hash: str | None = None
    result: list[tuple[str, bytes]] = []
    for tag in _ordered_grammar_tags():
        data = _tag_grammar(tag)
        if data is None:
            continue
        h = _hash(data)
        if h == seen_hash:
            continue
        seen_hash = h
        result.append((tag, data))
    return result


def _stage_and_commit(grammar_bytes: bytes, message: str, tag: str | None) -> None:
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    filename = _WORK_DIR / f"grammar-{tag or 'HEAD'}.js"
    filename.write_bytes(grammar_bytes)
    _run(["schema", "add", str(filename)], cwd=str(_VCS_ROOT))
    _run(["schema", "commit", "-m", message], cwd=str(_VCS_ROOT))
    if tag is not None:
        _run(["schema", "tag", tag], cwd=str(_VCS_ROOT))


def _reset_vcs() -> None:
    panproto_dir = _VCS_ROOT / ".panproto"
    if panproto_dir.exists():
        shutil.rmtree(panproto_dir)
    _run(["schema", "init"], cwd=str(_VCS_ROOT))


def _current_head_grammar() -> bytes:
    return (_REPO_ROOT / _GRAMMAR_PATH).read_bytes()


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

    revisions = _distinct_grammar_revisions()
    for tag, data in revisions:
        msg = f"qvr grammar at {tag}"
        print(f"  staging {tag} ({len(data)} bytes)", flush=True)
        _stage_and_commit(data, msg, tag)
        print(f"  committed and tagged {tag}", flush=True)

    head_bytes = _current_head_grammar()
    last_tag_data = revisions[-1][1] if revisions else b""
    if head_bytes != last_tag_data:
        msg = "qvr grammar at HEAD: homogenized surface"
        print(f"  staging HEAD ({len(head_bytes)} bytes)", flush=True)
        _stage_and_commit(head_bytes, msg, tag=None)
        print("  committed HEAD", flush=True)
    else:
        print("  HEAD identical to last tag, no new commit needed")

    print(f"\nfinal log:")
    log = _run(["schema", "log"], cwd=str(_VCS_ROOT))
    print(log.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
