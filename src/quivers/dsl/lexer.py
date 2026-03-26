"""Lexer for the quivers DSL (.qvr files)."""

from __future__ import annotations

from quivers.dsl.tokens import Token, TokenType, KEYWORDS


class LexError(Exception):
    """Raised when the lexer encounters an invalid character.

    Parameters
    ----------
    message : str
        Error description.
    line : int
        Source line number.
    col : int
        Source column number.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"line {line}, col {col}: {message}")


class Lexer:
    """Tokenizer for .qvr source text.

    Produces a stream of Token objects from raw source code.
    Handles comments (# to end of line), keywords, identifiers,
    integer literals, and all operators.

    Parameters
    ----------
    source : str
        The raw .qvr source text.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._bracket_depth = 0  # tracks nesting inside () and []

    def _peek(self) -> str:
        """Return the current character without advancing."""
        if self._pos >= len(self._source):
            return ""

        return self._source[self._pos]

    def _advance(self) -> str:
        """Consume and return the current character."""
        ch = self._source[self._pos]
        self._pos += 1

        if ch == "\n":
            self._line += 1
            self._col = 1

        else:
            self._col += 1

        return ch

    def _skip_whitespace_and_comments(self) -> list[Token]:
        """Skip spaces, tabs, and comments. Emit NEWLINE tokens."""
        newlines: list[Token] = []

        while self._pos < len(self._source):
            ch = self._peek()

            if ch == "#":
                # skip to end of line
                while self._pos < len(self._source) and self._peek() != "\n":
                    self._advance()

            elif ch == "\n":
                # suppress newlines inside balanced () and []
                if self._bracket_depth == 0:
                    newlines.append(
                        Token(TokenType.NEWLINE, "\\n", self._line, self._col)
                    )

                self._advance()

            elif ch in (" ", "\t", "\r"):
                self._advance()

            else:
                break

        return newlines

    def _read_ident(self) -> str:
        """Read an identifier or keyword."""
        start = self._pos

        while self._pos < len(self._source) and (
            self._source[self._pos].isalnum() or self._source[self._pos] == "_"
        ):
            self._advance()

        return self._source[start : self._pos]

    def _read_number(self) -> tuple[str, bool]:
        """Read an integer or float literal.

        Returns
        -------
        tuple[str, bool]
            The number string and whether it is a float.
        """
        start = self._pos
        is_float = False

        while self._pos < len(self._source) and self._source[self._pos].isdigit():
            self._advance()

        # check for decimal point followed by digits
        if (
            self._pos < len(self._source)
            and self._source[self._pos] == "."
            and self._pos + 1 < len(self._source)
            and self._source[self._pos + 1].isdigit()
        ):
            is_float = True
            self._advance()  # consume '.'

            while self._pos < len(self._source) and self._source[self._pos].isdigit():
                self._advance()

        return self._source[start : self._pos], is_float

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source into a list of tokens.

        Returns
        -------
        list[Token]
            The token stream, ending with an EOF token.

        Raises
        ------
        LexError
            If an unexpected character is encountered.
        """
        tokens: list[Token] = []

        while self._pos < len(self._source):
            # skip whitespace and comments, collect newlines
            newlines = self._skip_whitespace_and_comments()
            tokens.extend(newlines)

            if self._pos >= len(self._source):
                break

            line, col = self._line, self._col
            ch = self._peek()

            # handle <-, <<, and >=> operators
            if ch == "<":
                if self._pos + 1 < len(self._source) and self._source[self._pos + 1] == "-":
                    self._advance()
                    self._advance()
                    tokens.append(Token(TokenType.LARROW, "<-", line, col))
                elif self._pos + 1 < len(self._source) and self._source[self._pos + 1] == "<":
                    self._advance()
                    self._advance()
                    tokens.append(Token(TokenType.COMPOSE_BACK, "<<", line, col))
                else:
                    raise LexError(f"unexpected character {ch!r}", line, col)

            elif ch == "-" and self._pos + 1 < len(self._source) and self._source[self._pos + 1] == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.ARROW, "->", line, col))

            elif ch == "-":
                self._advance()
                tokens.append(Token(TokenType.MINUS, "-", line, col))

            elif ch == "/":
                self._advance()
                tokens.append(Token(TokenType.SLASH, "/", line, col))

            elif ch == "\\":
                self._advance()
                tokens.append(Token(TokenType.BACKSLASH, "\\", line, col))

            elif ch == ">" and self._pos + 1 < len(self._source) and self._source[self._pos + 1] == "=":
                if self._pos + 2 < len(self._source) and self._source[self._pos + 2] == ">":
                    self._advance()
                    self._advance()
                    self._advance()
                    tokens.append(Token(TokenType.KLEISLI, ">=>", line, col))
                else:
                    raise LexError(f"unexpected character sequence '>='", line, col)

            elif ch == ">" and self._pos + 1 < len(self._source) and self._source[self._pos + 1] == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.COMPOSE, ">>", line, col))

            # single-character operators
            elif ch == ":":
                self._advance()
                tokens.append(Token(TokenType.COLON, ":", line, col))

            elif ch == "*":
                self._advance()
                tokens.append(Token(TokenType.PRODUCT, "*", line, col))

            elif ch == "+":
                self._advance()
                tokens.append(Token(TokenType.COPRODUCT, "+", line, col))

            elif ch == "@":
                self._advance()
                tokens.append(Token(TokenType.TENSOR, "@", line, col))

            elif ch == "=" and self._pos + 1 < len(self._source) and self._source[self._pos + 1] == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.DARROW, "=>", line, col))

            elif ch == "=":
                self._advance()
                tokens.append(Token(TokenType.EQUALS, "=", line, col))

            elif ch == ".":
                self._advance()
                tokens.append(Token(TokenType.DOT, ".", line, col))

            elif ch == "(":
                self._advance()
                self._bracket_depth += 1
                tokens.append(Token(TokenType.LPAREN, "(", line, col))

            elif ch == ")":
                self._advance()
                if self._bracket_depth > 0:
                    self._bracket_depth -= 1

                tokens.append(Token(TokenType.RPAREN, ")", line, col))

            elif ch == "[":
                self._advance()
                self._bracket_depth += 1
                tokens.append(Token(TokenType.LBRACKET, "[", line, col))

            elif ch == "]":
                self._advance()
                if self._bracket_depth > 0:
                    self._bracket_depth -= 1

                tokens.append(Token(TokenType.RBRACKET, "]", line, col))

            elif ch == ",":
                self._advance()
                tokens.append(Token(TokenType.COMMA, ",", line, col))

            elif ch == "~":
                self._advance()
                tokens.append(Token(TokenType.TILDE, "~", line, col))

            # identifiers and keywords
            elif ch.isalpha() or ch == "_":
                word = self._read_ident()
                ttype = KEYWORDS.get(word, TokenType.IDENT)
                tokens.append(Token(ttype, word, line, col))

            # numeric literals (int or float)
            elif ch.isdigit():
                num, is_float = self._read_number()

                if is_float:
                    tokens.append(Token(TokenType.FLOAT, num, line, col))

                else:
                    tokens.append(Token(TokenType.INT, num, line, col))

            else:
                raise LexError(f"unexpected character {ch!r}", line, col)

        tokens.append(Token(TokenType.EOF, "", self._line, self._col))
        return tokens
