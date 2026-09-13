"""Fail when the checked-in VSIX differs from its authoritative sources."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "editors" / "vscode-qvr"
PARITY_FILES = (
    "package.json",
    "language-configuration.json",
    "syntaxes/qvr.tmLanguage.json",
    "out/extension.js",
    "out/extension.js.map",
)


def main(*, full: bool = False) -> int:
    manifest = json.loads((EXTENSION / "package.json").read_text())
    version = manifest["version"]
    expected = EXTENSION / f"vscode-qvr-{version}.vsix"
    archives = sorted(EXTENSION.glob("*.vsix"))
    if archives != [expected]:
        rendered = ", ".join(path.name for path in archives) or "none"
        sys.stderr.write(
            f"expected only {expected.name}; found {rendered}. "
            "Run `npm ci && npm run package` in editors/vscode-qvr.\n"
        )
        return 1
    failures: list[str] = []
    with ZipFile(expected) as archive:
        parity_files = list(PARITY_FILES)
        if full:
            parity_files.extend(
                name.removeprefix("extension/")
                for name in archive.namelist()
                if name.startswith("extension/node_modules/") and not name.endswith("/")
            )
        for relative in parity_files:
            archived = archive.read(f"extension/{relative}")
            current_path = EXTENSION / relative
            if not current_path.is_file():
                failures.append(f"{relative} (missing)")
                continue
            current = current_path.read_bytes()
            if archived != current:
                failures.append(relative)
    if failures:
        sys.stderr.write(
            f"{expected.name} has stale files: {', '.join(failures)}. "
            "Run `npm ci && npm run package` in editors/vscode-qvr.\n"
        )
        return 1
    print(f"OK {expected.name}: {len(parity_files)} source/archive files match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(full="--full" in sys.argv[1:]))
