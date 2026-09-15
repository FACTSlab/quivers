"""A griffe extension that gives didactic models their constructor.

Didactic models are built from annotated class attributes and constructed
by keyword, the way a keyword-only dataclass is, but they carry no
``@dataclass`` decorator for griffe's built-in extension to recognize. The
documentation therefore documents each field as a constructor parameter
in the class docstring, and this extension synthesizes the matching
``__init__`` so the parameters resolve against a signature.
"""

from __future__ import annotations

from griffe import (
    Attribute,
    Class,
    Expr,
    Extension,
    Function,
    GriffeLoader,
    Module,
    Parameter,
    ParameterKind,
    Parameters,
)

#: The canonical paths of the didactic bases every model descends from.
MODEL_BASES = frozenset(
    {
        "didactic.api.Model",
        "didactic.api.TaggedUnion",
        "didactic.Model",
        "didactic.TaggedUnion",
        "didactic.models._model.Model",
        "didactic.fields._unions.TaggedUnion",
    }
)


def _base_path(base: str | Expr) -> str:
    """Return a base expression's canonical path.

    Parameters
    ----------
    base : str | Expr
        A base as griffe records it on a class.

    Returns
    -------
    str
        The canonical dotted path, or the text of a base griffe could
        not resolve.
    """
    if isinstance(base, Expr):
        return base.canonical_path
    return base


def _is_model(class_: Class, seen: set[str]) -> bool:
    """Whether a class descends from a didactic model base.

    Parameters
    ----------
    class_ : Class
        The class to classify.
    seen : set[str]
        The paths already visited, which breaks a cycle of aliases.

    Returns
    -------
    bool
        True when a base is a didactic model base, or resolves within the
        loaded packages to a class that is itself a model.
    """
    if class_.path in seen:
        return False
    seen.add(class_.path)
    for base in class_.bases:
        path = _base_path(base)
        if path in MODEL_BASES:
            return True
        try:
            resolved = class_.modules_collection[path]
        except KeyError, AttributeError:
            continue
        if isinstance(resolved, Class) and _is_model(resolved, seen):
            return True
    return False


def _model_classes(class_: Class) -> list[Class]:
    """Return a class and its model ancestors, base first.

    Parameters
    ----------
    class_ : Class
        The class.

    Returns
    -------
    list[Class]
        Every class whose fields the model inherits, ancestors before
        descendants, so a redeclared field takes the descendant's
        annotation and default.
    """
    chain: list[Class] = []
    for base in class_.bases:
        path = _base_path(base)
        if path in MODEL_BASES:
            continue
        try:
            resolved = class_.modules_collection[path]
        except KeyError, AttributeError:
            continue
        if isinstance(resolved, Class):
            chain.extend(_model_classes(resolved))
    chain.append(class_)
    return chain


def _field_parameters(class_: Class) -> dict[str, Parameter]:
    """Return the constructor parameters a model class's own fields make.

    Parameters
    ----------
    class_ : Class
        The class.

    Returns
    -------
    dict[str, Parameter]
        One keyword-only parameter per annotated instance attribute, by
        name, in source order.
    """
    parameters: dict[str, Parameter] = {}
    for member in class_.members.values():
        if not isinstance(member, Attribute) or member.annotation is None:
            continue
        if "property" in member.labels:
            continue
        if (
            "class-attribute" in member.labels
            and "instance-attribute" not in member.labels
        ):
            continue
        parameters[member.name] = Parameter(
            member.name,
            annotation=member.annotation,
            kind=ParameterKind.keyword_only,
            default=member.value,
            docstring=member.docstring,
        )
    return parameters


def _set_model_init(class_: Class) -> None:
    """Give a model class the keyword-only constructor its fields imply.

    Parameters
    ----------
    class_ : Class
        The model class; it must not already define ``__init__``.
    """
    parameters: dict[str, Parameter] = {}
    for ancestor in _model_classes(class_):
        parameters.update(_field_parameters(ancestor))
    init = Function(
        "__init__",
        lineno=0,
        endlineno=0,
        parent=class_,
        parameters=Parameters(
            Parameter(
                name="self",
                annotation=None,
                kind=ParameterKind.positional_or_keyword,
                default=None,
            ),
            *parameters.values(),
        ),
        returns="None",
    )
    class_.set_member("__init__", init)
    class_.labels.add("didactic-model")


def _apply(node: Module | Class, processed: set[str]) -> None:
    """Synthesize constructors for every model class under a node.

    Parameters
    ----------
    node : Module | Class
        The module or class to descend into.
    processed : set[str]
        The paths already handled.
    """
    if node.canonical_path in processed:
        return
    processed.add(node.canonical_path)
    if isinstance(node, Class):
        if "__init__" not in node.members and _is_model(node, set()):
            _set_model_init(node)
        for member in node.members.values():
            if not member.is_alias and member.is_class:
                _apply(member, processed)  # type: ignore[arg-type]
    else:
        for member in node.members.values():
            if not member.is_alias and (member.is_module or member.is_class):
                _apply(member, processed)  # type: ignore[arg-type]


class DidacticModelsExtension(Extension):
    """Synthesize ``__init__`` for didactic models during static analysis."""

    def on_package(
        self, *, pkg: Module, loader: GriffeLoader, **kwargs: Module | GriffeLoader
    ) -> None:
        """Handle a loaded package.

        Parameters
        ----------
        pkg : Module
            The loaded package.
        loader : GriffeLoader
            The loader in use, which the extension does not need.
        **kwargs : Module | GriffeLoader
            Hook arguments a later griffe may add, which the extension
            ignores.
        """
        _apply(pkg, set())
