"""Hatch hook that compiles and bundles QVR native parsers."""

from __future__ import annotations

import json
import runpy
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Produce a platform wheel containing source-bound parser libraries."""

    PLUGIN_NAME = "custom"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._temporary_directories: list[Path] = []

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version != "standard":
            return

        root = Path(self.root)
        grammar_dir = root / "grammars" / "qvr"
        support = runpy.run_path(
            str(root / "src" / "quivers" / "dsl" / "_grammar_build.py")
        )
        extension = support["_shared_lib_extension"]()

        output_root = Path(
            tempfile.mkdtemp(prefix=".quivers-native-", dir=self.directory)
        )
        self._temporary_directories.append(output_root)
        force_include = build_data.setdefault("force_include", {})
        self._bundle_parser(
            support,
            grammar_dir=grammar_dir,
            output_dir=output_root / "current",
            library_name=f"qvr_grammar{extension}",
            manifest_name="manifest.json",
            package_dir="quivers/dsl/_grammar_native",
            force_include=force_include,
        )

        snapshots_root = grammar_dir / "vcs" / "parsers"
        snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
        for snapshot_dir in snapshots:
            source_dir = snapshot_dir / "source"
            if not (source_dir / "src" / "parser.c").is_file():
                continue
            name = snapshot_dir.name
            package_dir = f"quivers/cli/migrations/_grammar_vcs/parsers/{name}"
            force_include[str(source_dir / "src")] = f"{package_dir}/source/src"
            self._bundle_parser(
                support,
                grammar_dir=source_dir,
                output_dir=output_root / "migrations" / name,
                library_name=f"qvr{extension}",
                package_dir=package_dir,
                force_include=force_include,
            )
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    @staticmethod
    def _bundle_parser(
        support: dict[str, Any],
        *,
        grammar_dir: Path,
        output_dir: Path,
        library_name: str,
        manifest_name: str = "native-manifest.json",
        package_dir: str,
        force_include: dict[str, str],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        library = output_dir / library_name
        manifest = output_dir / manifest_name
        support["_compile_shared_lib_to"](grammar_dir, library)
        payload = support["_native_manifest_data"](grammar_dir, library)
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        force_include[str(library)] = f"{package_dir}/{library.name}"
        force_include[str(manifest)] = f"{package_dir}/{manifest.name}"

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        del version, build_data, artifact_path
        self._remove_temporary_directories()

    def clean(self, versions: list[str]) -> None:
        del versions
        self._remove_temporary_directories()
        build_dir = Path(self.directory)
        if build_dir.is_dir():
            for candidate in build_dir.glob(".quivers-native-*"):
                if candidate.is_dir():
                    shutil.rmtree(candidate)

    def _remove_temporary_directories(self) -> None:
        for directory in self._temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)
        self._temporary_directories.clear()
