"""The export policy of `docs/developer/public-api.md`, enforced.

Each test checks one rule of that page over every module of the
installed package, so a module that drifts from the policy fails here
by name rather than surfacing later as an import a downstream user
cannot make. The rendered-reference rule needs a built site and is
checked by `tools/check_api_reference.py` in the docs job instead.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import types
import typing
from pathlib import Path

import pytest
import torch
from torch import nn

import quivers
from quivers.continuous import ContinuousMorphism
from quivers.core import Morphism, extract_morphism
from quivers.dsl import load

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src" / "quivers"
_GALLERY = _ROOT / "docs" / "examples" / "source"

#: The target runtime sources, which are data read by path rather than
#: modules, per the policy's exceptions section.
_RUNTIME_PREFIX = "quivers.transpile.runtime_"


def _is_public(name: str) -> bool:
    return not any(part.startswith("_") for part in name.split("."))


def _module_names() -> list[str]:
    names = ["quivers"]
    for info in pkgutil.walk_packages(quivers.__path__, "quivers."):
        if info.name.startswith(_RUNTIME_PREFIX):
            continue
        names.append(info.name)
    return sorted(names)


_MODULES = _module_names()
_PUBLIC_MODULES = [name for name in _MODULES if _is_public(name)]


def _module(name: str) -> types.ModuleType:
    return importlib.import_module(name)


def _is_package(name: str) -> bool:
    return hasattr(_module(name), "__path__")


_PUBLIC_PACKAGES = [name for name in _PUBLIC_MODULES if _is_package(name)]
_PUBLIC_LEAVES = [name for name in _PUBLIC_MODULES if not _is_package(name)]


def _exports(name: str) -> list[str]:
    exported = getattr(_module(name), "__all__", None)
    assert exported is not None, f"{name} declares no __all__"
    return list(exported)


def _public_objects() -> dict[int, str]:
    """Every public object, keyed by identity, with one public path."""
    found: dict[int, str] = {}
    for package in _PUBLIC_PACKAGES:
        module = _module(package)
        for name in getattr(module, "__all__", ()):
            if hasattr(module, name):
                found.setdefault(id(getattr(module, name)), f"{package}.{name}")
    return found


@pytest.mark.parametrize("name", _MODULES)
def test_every_module_imports(name: str) -> None:
    """Rule 2 needs the module loaded; an import failure is a failure."""
    _module(name)


@pytest.mark.parametrize("name", _PUBLIC_MODULES)
def test_every_public_module_declares_all(name: str) -> None:
    """Rule 2: a public module states its public interface."""
    exported = getattr(_module(name), "__all__", None)
    assert exported is not None, f"{name} declares no __all__"
    assert isinstance(exported, (list, tuple)), f"{name}.__all__ is not a list"
    assert all(isinstance(entry, str) for entry in exported)


def _list_violations(module: types.ModuleType) -> list[str]:
    """Rule 4's violations in one module's `__all__`."""
    exported = list(module.__all__)
    problems: list[str] = []
    private = [entry for entry in exported if entry.startswith("_")]
    if private:
        problems.append(f"lists private names {private}")
    duplicates = sorted({entry for entry in exported if exported.count(entry) > 1})
    if duplicates:
        problems.append(f"repeats {duplicates}")
    unbound = [entry for entry in exported if not hasattr(module, entry)]
    if unbound:
        problems.append(f"lists unbound names {unbound}")
    return problems


def _reexport_violations(
    package: types.ModuleType, leaf: types.ModuleType
) -> list[str]:
    """Rule 3's violations between a package and one of its leaves."""
    package_exports = set(package.__all__)
    problems: list[str] = []
    missing = [entry for entry in leaf.__all__ if entry not in package_exports]
    if missing:
        problems.append(f"package omits {missing}")
    different = [
        entry
        for entry in leaf.__all__
        if entry in package_exports
        and getattr(package, entry, None) is not getattr(leaf, entry)
    ]
    if different:
        problems.append(f"package binds {different} to other objects")
    return problems


@pytest.mark.parametrize("name", _PUBLIC_MODULES)
def test_all_lists_resolvable_public_names_once(name: str) -> None:
    """Rule 4: no private names, no duplicates, and every name binds."""
    problems = _list_violations(_module(name))
    assert not problems, f"{name}.__all__ {'; '.join(problems)}"


@pytest.mark.parametrize("name", _PUBLIC_LEAVES)
def test_packages_reexport_their_leaf_modules(name: str) -> None:
    """Rule 3: a leaf's exports are its package's, as the same objects."""
    package_name = name.rpartition(".")[0]
    problems = _reexport_violations(_module(package_name), _module(name))
    assert not problems, f"{package_name} against {name}: {'; '.join(problems)}"


def _quivers_types(
    annotation: object, found: set[type], seen: set[int] | None = None
) -> None:
    """Collect the Quivers classes an annotation mentions."""
    seen = set() if seen is None else seen
    if id(annotation) in seen:
        return
    seen.add(id(annotation))
    if isinstance(annotation, typing.TypeVar):
        if annotation.__bound__ is not None:
            _quivers_types(annotation.__bound__, found, seen)
        for constraint in annotation.__constraints__:
            _quivers_types(constraint, found, seen)
        return
    if isinstance(annotation, typing.TypeAliasType):
        _quivers_types(annotation.__value__, found, seen)
        return
    origin = typing.get_origin(annotation)
    if origin is None and isinstance(annotation, type):
        found.add(annotation)
        return
    if isinstance(origin, type):
        found.add(origin)
    for argument in typing.get_args(annotation):
        if isinstance(argument, (list, tuple)):
            for item in argument:
                _quivers_types(item, found, seen)
        else:
            _quivers_types(argument, found, seen)


def _signature_sites() -> list[tuple[str, object]]:
    """Every public function, constructor, and public method."""
    sites: dict[int, tuple[str, object]] = {}
    for path in _public_objects().values():
        package, _, name = path.rpartition(".")
        value = getattr(_module(package), name)
        if not getattr(value, "__module__", "").startswith("quivers"):
            continue
        if inspect.isfunction(value):
            sites.setdefault(id(value), (path, value))
        elif inspect.isclass(value):
            for attribute, member in vars(value).items():
                public_method = not attribute.startswith("_")
                if attribute == "__init__" or public_method:
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        sites.setdefault(id(member), (f"{path}.{attribute}", member))
    return sorted(sites.values(), key=lambda site: site[0])


_SITES = _signature_sites()


def _internal_types(site: object, public: dict[int, str], prefix: str) -> list[str]:
    """Rule 5's violations: internal types a signature annotates."""
    found: set[type] = set()
    for annotation in typing.get_type_hints(site).values():
        _quivers_types(annotation, found)
    return sorted(
        f"{kind.__module__}.{kind.__qualname__}"
        for kind in found
        if kind.__module__.startswith(prefix) and id(kind) not in public
    )


@pytest.mark.parametrize(
    "site", [site[1] for site in _SITES], ids=[site[0] for site in _SITES]
)
def test_public_signatures_name_public_types(site: object) -> None:
    """Rule 5: every annotated Quivers type of a public call is public."""
    internal = _internal_types(site, _public_objects(), "quivers")
    assert not internal, f"public signature names internal types {internal}"


def _morphism_classes(root: nn.Module) -> set[type]:
    """The morphism classes reachable from a compiled program."""
    found: set[type] = set()
    for module in root.modules():
        if isinstance(module, (ContinuousMorphism, Morphism)):
            found.add(type(module))
        wrapped = extract_morphism(module)
        if wrapped is not None:
            found.add(type(wrapped))
    return found


@pytest.mark.parametrize("stem", sorted(p.stem for p in _GALLERY.glob("*.qvr")))
def test_compiled_programs_contain_public_morphisms(stem: str) -> None:
    """Rule 6: Python can build every morphism the compiler builds."""
    program = load(str(_GALLERY / f"{stem}.qvr"))
    public = _public_objects()
    internal = sorted(
        f"{kind.__module__}.{kind.__qualname__}"
        for kind in _morphism_classes(program)
        if id(kind) not in public
    )
    assert not internal, f"{stem} compiles to internal morphism classes {internal}"


def test_program_step_records_are_public() -> None:
    """Rule 6: the records a `MonadicProgram` is built from are public."""
    public = _public_objects()
    hints = typing.get_type_hints(quivers.continuous.MonadicProgram.__init__)
    found: set[type] = set()
    _quivers_types(hints["steps"], found)
    assert found, "MonadicProgram's steps parameter names no step type"
    for kind in found:
        if kind.__module__.startswith("quivers"):
            assert id(kind) in public, f"{kind.__qualname__} is not public"


def test_runtime_sources_are_not_imported() -> None:
    """The exceptions section: no module imports a target runtime source."""
    offenders: list[str] = []
    for path in sorted(_SOURCE.rglob("*.py")):
        if path.name.startswith("runtime_") and path.parent.name == "transpile":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            if any(name.startswith(_RUNTIME_PREFIX) for name in imported):
                offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"modules import a target runtime source: {offenders}"


class _ProbeInternal:
    """A type no public package exports, for the rule-5 mutation test."""


def _probe_uses_internal(value: _ProbeInternal) -> list[torch.Tensor]:
    """A signature naming an unexported type.

    Parameters
    ----------
    value : _ProbeInternal
        Unused.

    Returns
    -------
    list[torch.Tensor]
        A single zero.
    """
    del value
    return [torch.zeros(())]


def test_rule_checks_reject_violations() -> None:
    """The checks above fail on the violations they exist to catch."""
    leaf = types.ModuleType("probe.leaf")
    leaf.shown = object()
    leaf.__all__ = ["_hidden", "shown", "shown", "absent"]
    problems = _list_violations(leaf)
    assert any("private names ['_hidden']" in p for p in problems)
    assert any("repeats ['shown']" in p for p in problems)
    assert any("unbound names" in p and "'absent'" in p for p in problems)
    assert not any("unbound names" in p and "'shown'" in p for p in problems)

    leaf.__all__ = ["shown", "other"]
    leaf.other = object()
    package = types.ModuleType("probe")
    package.shown = object()
    package.__all__ = ["shown"]
    problems = _reexport_violations(package, leaf)
    assert any("omits ['other']" in p for p in problems)
    assert any("binds ['shown']" in p for p in problems)

    assert _internal_types(_probe_uses_internal, {}, _ProbeInternal.__module__) == [
        f"{_ProbeInternal.__module__}.{_ProbeInternal.__qualname__}"
    ]
    assert (
        _internal_types(
            _probe_uses_internal,
            {id(_ProbeInternal): "probe._ProbeInternal"},
            _ProbeInternal.__module__,
        )
        == []
    )
