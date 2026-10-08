# Overview

The pipeline from a compiled module to a target program, with the public
entry points. Every public name of the transpile package is importable
from `quivers.transpile`; the names its leaf modules define are
documented on their own pages (plan, IR, kernel IR, family metadata,
family spelling, renderer registry), and the renderers on the renderers
page. The constants and type aliases those modules define are listed here
under their package path.

::: quivers.transpile
    options:
      members:
        - transpile
        - available_targets
        - UnsupportedConstruct
        - RefusedDeclaration
        - unsupported_for
        - Backend
        - STAN_LIKE
        - PYTHON_DEEP
        - CHURCH_LIKE
        - CATEGORICAL_METADATA_IGNORABLE
        - QIEC_SURFACE
        - EmitPretty
        - parser_registry
        - target_protocol
        - FAMILY_META
        - MarginalizeAtomSet
        - DYNAMIC_TARGET_LANGUAGES
        - TARGET_NAME_SEPARATOR
        - QiecFeature
        - EmitFn
