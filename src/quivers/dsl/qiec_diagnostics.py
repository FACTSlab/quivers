"""The diagnostic the QIEC elaboration raises for a rejected source."""

from __future__ import annotations


class QiecDiagnosticError(ValueError):
    """A source-located, stable-code diagnostic from QIEC elaboration.

    Parameters
    ----------
    message : str
        What was rejected.
    code : str
        The stable diagnostic code.
    file : str
        The file the source was read from.
    line : int
        The one-based line, or zero when unknown.
    column : int
        The zero-based column.
    program : str | None
        The program whose elaboration raised the diagnostic, when it was
        raised inside a program body; ``None`` for any other declaration.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        file: str,
        line: int = 0,
        column: int = 0,
        program: str | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.file = file
        self.line = line
        self.column = column
        self.program = program
        location = f"{file}:{line}:{column}" if line else file
        super().__init__(f"{location}: [{code}] {message}")


__all__ = ["QiecDiagnosticError"]
