"""Check that every public name renders in the built API reference.

Rule 7 of `docs/developer/public-api.md`: each name a public package
lists in `__all__` has an anchor in the site `mkdocs build` writes,
either at a public path (``quivers.continuous.PlateDraw``) or at the
path of its definition (``quivers.continuous.plate.PlateDraw``). The
documentation generator omits an object without a docstring, so a
missing anchor usually means a missing docstring or a missing `:::`
directive.

Usage::

    python tools/check_api_reference.py site
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import re
import sys
from pathlib import Path

import quivers

_ANCHOR = re.compile(r'id="(quivers(?:\.[A-Za-z0-9_]+)+)"')
_RUNTIME_PREFIX = "quivers.transpile.runtime_"


def _is_public(name: str) -> bool:
    """Whether no component of a dotted module path is private.

    Parameters
    ----------
    name : str
        A dotted module path.

    Returns
    -------
    bool
        ``True`` when no component begins with an underscore.
    """
    return not any(part.startswith("_") for part in name.split("."))


def public_modules(*, packages_only: bool) -> list[str]:
    """The public modules of the installed library.

    Parameters
    ----------
    packages_only : bool
        Whether to list packages only, leaving out leaf modules.

    Returns
    -------
    list[str]
        Dotted paths of every module with no private component.
    """
    names = ["quivers"]
    for info in pkgutil.walk_packages(quivers.__path__, "quivers."):
        if info.name.startswith(_RUNTIME_PREFIX):
            continue
        if info.ispkg or not packages_only:
            names.append(info.name)
    return sorted(name for name in names if _is_public(name))


def public_packages() -> list[str]:
    """The public packages of the installed library.

    Returns
    -------
    list[str]
        Dotted paths of every package with no private component.
    """
    return public_modules(packages_only=True)


def public_paths() -> dict[int, set[str]]:
    """Every public path of every public object.

    A leaf module's path counts as well as a package's, so a constant
    documented on the page of the module that defines it is found.

    Returns
    -------
    dict[int, set[str]]
        The public paths each object is bound at, keyed by identity, so
        a name re-exported by several modules has all of its paths.
    """
    paths: dict[int, set[str]] = {}
    for package in public_modules(packages_only=False):
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            paths.setdefault(id(value), set()).add(f"{package}.{name}")
    return paths


def accepted_anchors(package: str, name: str, paths: dict[int, set[str]]) -> set[str]:
    """The anchors any one of which documents a public name.

    Parameters
    ----------
    package : str
        The public package listing the name.
    name : str
        The listed name.
    paths : dict[int, set[str]]
        Every public path of every public object, from `public_paths`.

    Returns
    -------
    set[str]
        Every public path the object is bound at, and its definition
        path when the object records one.
    """
    value = getattr(importlib.import_module(package), name)
    anchors = {f"{package}.{name}", *paths.get(id(value), set())}
    home = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(home, str) and isinstance(qualname, str):
        anchors.add(f"{home}.{qualname}")
    return anchors


def site_anchors(site: Path) -> set[str]:
    """Every Quivers anchor in a built site.

    Parameters
    ----------
    site : Path
        The directory `mkdocs build` wrote.

    Returns
    -------
    set[str]
        The ``id`` attributes that name a Quivers object.
    """
    anchors: set[str] = set()
    for page in site.rglob("*.html"):
        anchors.update(_ANCHOR.findall(page.read_text(errors="replace")))
    return anchors


def undocumented(site: Path) -> list[str]:
    """The public names with no anchor in a built site.

    Parameters
    ----------
    site : Path
        The directory `mkdocs build` wrote.

    Returns
    -------
    list[str]
        Public paths of the names the site does not document.
    """
    anchors = site_anchors(site)
    paths = public_paths()
    missing: list[str] = []
    for package in public_packages():
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            if not accepted_anchors(package, name, paths) & anchors:
                missing.append(f"{package}.{name}")
    return missing


def main() -> int:
    """Report undocumented public names and fail when there are any.

    Returns
    -------
    int
        The process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Check that every public name renders in the API reference."
    )
    parser.add_argument("site", type=Path, help="the built documentation site")
    arguments = parser.parse_args()
    if not (arguments.site / "index.html").is_file():
        print(f"{arguments.site} holds no built site", file=sys.stderr)
        return 2
    missing = undocumented(arguments.site)
    for path in missing:
        print(f"undocumented public name: {path}")
    if missing:
        print(f"{len(missing)} public names have no API reference anchor")
        return 1
    print("every public name renders in the API reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
