"""``qvr run``: execute one named, checked QIEC computation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from quivers.dsl import Compiler, ParseError, parse
from quivers.qiec import (
    ExecutionDiagnostic,
    ExecutionFailure,
    RuntimeConfiguration,
    load_runtime_configuration,
    parse_static_arguments,
    run_named,
)


def _argument(text: str) -> object:
    try:
        return _tuples(json.loads(text))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON value argument {text!r}: {error.msg}"
        ) from error


def _tuples(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _emit_failure(
    diagnostic: ExecutionDiagnostic,
    *,
    json_output: bool,
) -> int:
    if json_output:
        sys.stdout.write(
            json.dumps({"ok": False, "diagnostics": [diagnostic.to_data()]}, indent=2)
            + "\n"
        )
    else:
        origin = diagnostic.origin
        location = ""
        if origin is not None and origin.file:
            location = origin.file
            if origin.line is not None:
                location += f":{origin.line}:{origin.column or 0}"
            location += ": "
        sys.stderr.write(
            f"{location}{diagnostic.severity}[{diagnostic.code}]: "
            f"{diagnostic.message}\n"
        )
    return 1


def main(args: object) -> int:
    """Execute parsed argparse ``run`` arguments."""

    path = Path(str(getattr(args, "file")))
    json_output = bool(getattr(args, "json", False))
    computation = str(getattr(args, "computation"))
    try:
        source = path.read_bytes()
        parsed = parse(source, file_path=str(path))
        compiler = Compiler(parsed, module_name=path.stem, file_path=str(path))
        compiler.compile()
        module = compiler.qiec_module
        if module is None:
            raise ExecutionFailure(
                ExecutionDiagnostic(
                    "qiec-run-module",
                    f"{path} contains no QIEC declarations",
                    computation,
                )
            )
        runtime_path = getattr(args, "runtime", None)
        runtime = (
            load_runtime_configuration(runtime_path)
            if runtime_path
            else RuntimeConfiguration()
        )
        static_arguments = parse_static_arguments(
            module,
            computation,
            tuple(getattr(args, "static", ()) or ()),
        )
        arguments = tuple(_argument(item) for item in getattr(args, "arguments", ()))
        fuel = getattr(args, "fuel", None)
        if fuel is not None and fuel <= 0:
            raise ValueError("--fuel must be a positive number of steps")
        result = run_named(
            module,
            computation,
            arguments,
            static_arguments=static_arguments,
            runtime=runtime,
            fuel=fuel,
        )
    except ExecutionFailure as error:
        return _emit_failure(error.diagnostic, json_output=json_output)
    except ParseError as error:
        return _emit_failure(
            ExecutionDiagnostic("parse", str(error), computation),
            json_output=json_output,
        )
    except (OSError, ValueError) as error:
        return _emit_failure(
            ExecutionDiagnostic("qiec-run-config", str(error), computation),
            json_output=json_output,
        )
    except Exception as error:
        return _emit_failure(
            ExecutionDiagnostic("compile", str(error), computation),
            json_output=json_output,
        )

    if json_output:
        sys.stdout.write(json.dumps(result.to_data(), indent=2) + "\n")
    else:
        sys.stdout.write(
            f"{result.computation} = {result.to_data()['value']!r} "
            f": {result.to_data()['result_type']}\n"
        )
        if bool(getattr(args, "trace", False)):
            for event in result.trace:
                sys.stderr.write(
                    f"trace[{event.sequence}] {event.event} "
                    f"{json.dumps(dict(event.detail), default=repr, sort_keys=True)}\n"
                )
    return 0


__all__ = ["main"]
