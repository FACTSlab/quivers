"""Recursive descent parser for the quivers DSL.

Parses a token stream into an AST (Module of Statements).

Grammar
-------
::

    program        := statement*
    statement      := quantale_decl | category_decl | rule_decl
                    | object_decl | morphism_decl
                    | let_decl | output_decl
                    | space_decl | continuous_decl | stochastic_decl
                    | discretize_decl | embed_decl
                    | program_decl

    category_decl  := 'category' IDENT (',' IDENT)*
    rule_decl      := 'rule' IDENT '(' IDENT (',' IDENT)* ')'
                      ':' cat_pattern (',' cat_pattern)* '=>' cat_pattern
    cat_pattern    := cat_slash
    cat_slash      := cat_product (('/' | '\\') cat_product)*
    cat_product    := cat_primary ('*' cat_primary)*
    cat_primary    := IDENT | '(' cat_pattern ')'

    quantale_decl  := 'quantale' IDENT
    object_decl    := 'object' IDENT ':' type_expr
    morphism_decl  := ('latent' | 'observed') IDENT ':' type_expr
                      '->' type_expr ['[' options ']'] ['=' expr]
    options        := IDENT '=' (IDENT | INT | FLOAT) (',' IDENT '=' ...)*
    let_decl       := 'let' IDENT '=' expr
    output_decl    := 'output' expr

    space_decl     := 'space' IDENT ':' space_expr
    space_expr     := space_product
    space_product  := space_primary ('*' space_primary)*
    space_primary  := IDENT '(' space_args ')' | IDENT

    space_args     := space_arg (',' space_arg)*
    space_arg      := IDENT '=' (INT | FLOAT) | INT | FLOAT

    continuous_decl := 'continuous' IDENT ':' IDENT '->' IDENT
                       '~' IDENT ['[' options ']']
    stochastic_decl := 'stochastic' IDENT ':' type_expr '->' type_expr
    discretize_decl := 'discretize' IDENT ':' IDENT '->' INT
    embed_decl      := 'embed' IDENT ':' IDENT '->' IDENT

    program_decl   := 'program' IDENT ['(' param_list ')'] ':' type_expr '->' type_expr
                       program_step+ return_stmt
    param_list     := IDENT (',' IDENT)*
    program_step   := draw_step | observe_step | let_step
    draw_step      := 'draw' var_pattern '~' IDENT ['(' arg_list ')']
    observe_step   := 'observe' var_pattern '~' IDENT ['(' arg_list ')']
    let_step       := 'let' IDENT '=' (IDENT | INT | FLOAT)
    var_pattern    := IDENT | '(' IDENT (',' IDENT)* ')'
    arg_list       := draw_arg (',' draw_arg)*
    draw_arg       := IDENT | INT | FLOAT
    return_stmt    := 'return' return_pattern
    return_pattern := IDENT | '(' return_entry (',' return_entry)* ')'
    return_entry   := IDENT ':' IDENT | IDENT

    type_expr      := coproduct_type
    coproduct_type := product_type ('+' product_type)*
    product_type   := primary_type ('*' primary_type)*
    primary_type   := IDENT | INT | '(' type_expr ')'

    expr           := compose_expr
    compose_expr   := tensor_expr ('>>' tensor_expr)*
    tensor_expr    := postfix_expr ('@' postfix_expr)*
    postfix_expr   := atom_expr ('.' method_call)*
    method_call    := 'marginalize' '(' IDENT (',' IDENT)* ')'
    atom_expr      := 'identity' '(' IDENT ')'
                    | IDENT
                    | '(' expr ')'
"""

from __future__ import annotations

from quivers.dsl.tokens import Token, TokenType
from quivers.dsl.ast_nodes import (
    Module,
    Statement,
    QuantaleDecl,
    CategoryDecl,
    RuleDecl,
    CatPatternName,
    CatPatternSlash,
    CatPatternProduct,
    CatPattern,
    ObjectDecl,
    MorphismDecl,
    SpaceDecl,
    ContinuousMorphismDecl,
    StochasticMorphismDecl,
    DiscretizeDecl,
    EmbedDecl,
    DrawStep,
    LetStep,
    LetExprBinOp,
    LetExprUnaryOp,
    LetExprCall,
    LetExprLiteral,
    LetExprVar,
    LetExprNode,
    ProgramDecl,
    LetDecl,
    OutputDecl,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeCoproduct,
    SpaceExpr,
    SpaceConstructor,
    SpaceProduct,
    Expr,
    ExprIdent,
    ExprIdentity,
    ExprCompose,
    ExprTensorProduct,
    ExprMarginalize,
    ExprFan,
    ExprRepeat,
    ExprStack,
    ExprScan,
    ExprParser,
)

# built-in functions available in let expressions
_LET_BUILTINS = {"sigmoid", "exp", "log", "abs", "softplus"}


class ParseError(Exception):
    """Raised when the parser encounters unexpected tokens.

    Parameters
    ----------
    message : str
        Error description.
    token : Token
        The offending token.
    """

    def __init__(self, message: str, token: Token) -> None:
        self.token = token
        super().__init__(
            f"line {token.line}, col {token.col}: {message} "
            f"(got {token.type.name} {token.value!r})"
        )


class Parser:
    """Recursive descent parser for .qvr programs.

    Parameters
    ----------
    tokens : list[Token]
        The token stream from the lexer.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _current(self) -> Token:
        """Return the current token."""
        return self._tokens[self._pos]

    def _peek_type(self) -> TokenType:
        """Return the type of the current token."""
        return self._tokens[self._pos].type

    def _peek_ahead(self, offset: int = 1) -> TokenType:
        """Look ahead by offset tokens, skipping nothing."""
        idx = self._pos + offset

        if idx < len(self._tokens):
            return self._tokens[idx].type

        return TokenType.EOF

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        """Consume a token of the expected type, or raise ParseError."""
        tok = self._current()

        if tok.type != ttype:
            raise ParseError(f"expected {ttype.name}", tok)

        return self._advance()

    def _skip_newlines(self) -> None:
        """Skip over any NEWLINE tokens."""
        while self._pos < len(self._tokens) and self._peek_type() == TokenType.NEWLINE:
            self._advance()

    def _at_statement_boundary(self) -> bool:
        """Check if we're at a statement start or EOF."""
        return self._peek_type() in (
            TokenType.QUANTALE,
            TokenType.OBJECT,
            TokenType.LATENT,
            TokenType.OBSERVED,
            TokenType.LET,
            TokenType.OUTPUT,
            TokenType.SPACE,
            TokenType.CONTINUOUS,
            TokenType.STOCHASTIC,
            TokenType.DISCRETIZE,
            TokenType.EMBED,
            TokenType.PROGRAM,
            TokenType.TYPE,
            TokenType.WHERE,
            TokenType.EOF,
        )

    # -- top-level -----------------------------------------------------------

    def parse(self) -> Module:
        """Parse the entire token stream into a Module.

        Returns
        -------
        Module
            The parsed AST.

        Raises
        ------
        ParseError
            On syntax errors.
        """
        statements: list[Statement] = []
        self._skip_newlines()

        while self._peek_type() != TokenType.EOF:
            result = self._parse_statement()

            if isinstance(result, list):
                statements.extend(result)

            else:
                statements.append(result)

            self._skip_newlines()

        return Module(statements=statements)

    def _parse_statement(self) -> Statement | list[Statement]:
        """Parse a single top-level statement."""
        ttype = self._peek_type()

        if ttype == TokenType.QUANTALE:
            return self._parse_quantale_decl()

        elif ttype == TokenType.CATEGORY:
            return self._parse_category_decl()

        elif ttype == TokenType.RULE:
            return self._parse_rule_decl()

        elif ttype == TokenType.OBJECT:
            return self._parse_object_decl()

        elif ttype in (TokenType.LATENT, TokenType.OBSERVED):
            return self._parse_morphism_decl()

        elif ttype == TokenType.LET:
            return self._parse_let_decl()

        elif ttype == TokenType.OUTPUT:
            return self._parse_output_decl()

        elif ttype == TokenType.SPACE:
            return self._parse_space_decl()

        elif ttype == TokenType.CONTINUOUS:
            return self._parse_continuous_decl()

        elif ttype == TokenType.STOCHASTIC:
            return self._parse_stochastic_decl()

        elif ttype == TokenType.DISCRETIZE:
            return self._parse_discretize_decl()

        elif ttype == TokenType.EMBED:
            return self._parse_embed_decl()

        elif ttype == TokenType.PROGRAM:
            return self._parse_program_decl()

        elif ttype == TokenType.TYPE:
            return self._parse_space_decl_from_type()

        else:
            raise ParseError("expected statement", self._current())

    # -- declarations --------------------------------------------------------

    def _parse_quantale_decl(self) -> QuantaleDecl:
        """Parse: quantale <name>."""
        tok = self._expect(TokenType.QUANTALE)
        name_tok = self._expect(TokenType.IDENT)
        return QuantaleDecl(name=name_tok.value, line=tok.line, col=tok.col)

    def _parse_category_decl(self) -> CategoryDecl | list[CategoryDecl]:
        """Parse: category <name> [, <name> ...].

        Supports both single and comma-separated declarations::

            category S
            category S, NP, N, VP, PP
        """
        tok = self._expect(TokenType.CATEGORY)
        name_tok = self._expect(TokenType.IDENT)
        first = CategoryDecl(name=name_tok.value, line=tok.line, col=tok.col)

        if self._peek_type() != TokenType.COMMA:
            return first

        decls = [first]

        while self._peek_type() == TokenType.COMMA:
            self._advance()
            name_tok = self._expect(TokenType.IDENT)
            decls.append(
                CategoryDecl(name=name_tok.value, line=tok.line, col=tok.col)
            )

        return decls

    def _parse_rule_decl(self) -> RuleDecl:
        """Parse: rule <name>(<var>, ...) : <premise>, ... => <conclusion>.

        Category patterns support ``/`` (forward slash), ``\\``
        (backward slash), ``*`` (tensor product), and parenthesized
        subexpressions. Names in the variable list are universally
        quantified; all other names are category atom constants.
        """
        tok = self._expect(TokenType.RULE)
        name_tok = self._expect(TokenType.IDENT)

        # parse parameter list: (X, Y, ...)
        self._expect(TokenType.LPAREN)
        variables: list[str] = []
        var_tok = self._expect(TokenType.IDENT)
        variables.append(var_tok.value)

        while self._peek_type() == TokenType.COMMA:
            self._advance()
            var_tok = self._expect(TokenType.IDENT)
            variables.append(var_tok.value)

        self._expect(TokenType.RPAREN)

        self._expect(TokenType.COLON)

        # parse premises: cat_pattern (',' cat_pattern)*
        premises: list[CatPattern] = []
        premises.append(self._parse_cat_pattern())

        while self._peek_type() == TokenType.COMMA:
            self._advance()
            premises.append(self._parse_cat_pattern())

        # expect =>
        self._expect(TokenType.DARROW)

        # parse conclusion
        conclusion = self._parse_cat_pattern()

        return RuleDecl(
            name=name_tok.value,
            variables=tuple(variables),
            premises=tuple(premises),
            conclusion=conclusion,
            line=tok.line,
            col=tok.col,
        )

    def _parse_cat_pattern(self) -> CatPattern:
        """Parse a category pattern (for rule declarations).

        Grammar::

            cat_pattern   := cat_slash
            cat_slash     := cat_primary (('/' | '\\\\') cat_primary)*
            cat_primary   := IDENT
                           | '(' cat_pattern ')'
                           | cat_primary '*' cat_primary

        Slash associates to the right: ``X/Y/Z`` = ``X/(Y/Z)``.
        Product binds tighter than slash.
        """
        return self._parse_cat_slash()

    def _parse_cat_slash(self) -> CatPattern:
        """Parse slash-level category pattern (right-associative)."""
        left = self._parse_cat_product()

        while self._peek_type() in (TokenType.SLASH, TokenType.BACKSLASH):
            dir_tok = self._current()
            direction = dir_tok.value
            self._advance()
            right = self._parse_cat_product()
            left = CatPatternSlash(
                result=left,
                argument=right,
                direction=direction,
                line=dir_tok.line,
                col=dir_tok.col,
            )

        return left

    def _parse_cat_product(self) -> CatPattern:
        """Parse product-level category pattern."""
        left = self._parse_cat_primary()

        while self._peek_type() == TokenType.PRODUCT:
            self._advance()
            right = self._parse_cat_primary()
            left = CatPatternProduct(
                left=left,
                right=right,
                line=left.line,
                col=left.col,
            )

        return left

    def _parse_cat_primary(self) -> CatPattern:
        """Parse primary category pattern: IDENT or parenthesized."""
        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            pat = self._parse_cat_pattern()
            self._expect(TokenType.RPAREN)
            return pat

        tok = self._expect(TokenType.IDENT)
        return CatPatternName(name=tok.value, line=tok.line, col=tok.col)

    def _parse_object_decl(self) -> ObjectDecl:
        """Parse: object <name> : <type_expr>."""
        tok = self._expect(TokenType.OBJECT)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.COLON)
        texpr = self._parse_type_expr()
        return ObjectDecl(
            name=name_tok.value, type_expr=texpr, line=tok.line, col=tok.col
        )

    def _parse_morphism_decl(self) -> MorphismDecl:
        """Parse: latent|observed <name> : <dom> -> <cod> [opts] [= expr]."""
        kind_tok = self._advance()
        kind = kind_tok.value
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.COLON)
        domain = self._parse_type_expr()
        self._expect(TokenType.ARROW)
        codomain = self._parse_type_expr()

        # optional [key=val, ...] options
        options: dict[str, str] = {}

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            options = self._parse_options()
            self._expect(TokenType.RBRACKET)

        # optional = expr
        init_expr = None

        if self._peek_type() == TokenType.EQUALS:
            self._advance()
            init_expr = self._parse_expr()

        return MorphismDecl(
            kind=kind,
            name=name_tok.value,
            domain=domain,
            codomain=codomain,
            init_expr=init_expr,
            options=options,
            line=kind_tok.line,
            col=kind_tok.col,
        )

    def _parse_options(self) -> dict[str, str]:
        """Parse key=value pairs inside brackets."""
        opts: dict[str, str] = {}
        key_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.EQUALS)
        val_tok = self._advance()
        opts[key_tok.value] = val_tok.value

        while self._peek_type() == TokenType.COMMA:
            self._advance()
            key_tok = self._expect(TokenType.IDENT)
            self._expect(TokenType.EQUALS)
            val_tok = self._advance()
            opts[key_tok.value] = val_tok.value

        return opts

    def _parse_let_decl(self) -> LetDecl:
        """Parse: let <name> = <expr> [where let_decl+]."""
        tok = self._expect(TokenType.LET)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.EQUALS)
        expr = self._parse_expr()

        # check for where clause
        self._skip_newlines()
        where_bindings = []
        if self._peek_type() == TokenType.WHERE:
            self._advance()
            self._skip_newlines()
            while self._peek_type() == TokenType.LET:
                where_bindings.append(self._parse_let_decl())
                self._skip_newlines()

        return LetDecl(
            name=name_tok.value, expr=expr,
            where=tuple(where_bindings) if where_bindings else None,
            line=tok.line, col=tok.col
        )

    def _parse_output_decl(self) -> OutputDecl:
        """Parse: output <expr>."""
        tok = self._expect(TokenType.OUTPUT)
        expr = self._parse_expr()
        return OutputDecl(expr=expr, line=tok.line, col=tok.col)

    # -- space declarations --------------------------------------------------

    def _parse_space_decl(self) -> SpaceDecl:
        """Parse: space <name> : <space_expr>."""
        tok = self._expect(TokenType.SPACE)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.COLON)
        sexpr = self._parse_space_expr()
        return SpaceDecl(
            name=name_tok.value, space_expr=sexpr,
            line=tok.line, col=tok.col,
        )

    def _parse_space_decl_from_type(self) -> SpaceDecl:
        """Parse: type <name> = <space_expr>.

        Alternative syntax using = instead of : (ML-style).
        """
        tok = self._expect(TokenType.TYPE)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.EQUALS)
        sexpr = self._parse_space_expr()
        return SpaceDecl(
            name=name_tok.value, space_expr=sexpr,
            line=tok.line, col=tok.col,
        )

    def _parse_space_expr(self) -> SpaceExpr:
        """Parse a space expression (product is lowest precedence)."""
        return self._parse_space_product()

    def _parse_space_product(self) -> SpaceExpr:
        """Parse: space_primary ('*' space_primary)*."""
        left = self._parse_space_primary()
        components = [left]

        while self._peek_type() == TokenType.PRODUCT:
            self._advance()
            components.append(self._parse_space_primary())

        if len(components) == 1:
            return components[0]

        return SpaceProduct(
            components=tuple(components),
            line=getattr(components[0], "line", 0),
            col=getattr(components[0], "col", 0),
        )

    def _parse_space_primary(self) -> SpaceExpr:
        """Parse: IDENT '(' args ')' | IDENT INT | IDENT FLOAT | IDENT."""
        tok = self._expect(TokenType.IDENT)

        # check if followed by '(' -> constructor call with parens
        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            args, kwargs = self._parse_space_args()
            self._expect(TokenType.RPAREN)
            return SpaceConstructor(
                constructor=tok.value,
                args=tuple(args),
                kwargs=kwargs,
                line=tok.line,
                col=tok.col,
            )

        # check if followed by INT/FLOAT -> parens-free constructor
        if self._peek_type() in (TokenType.INT, TokenType.FLOAT):
            arg_tok = self._advance()
            return SpaceConstructor(
                constructor=tok.value,
                args=(arg_tok.value,),
                kwargs={},
                line=tok.line,
                col=tok.col,
            )

        # bare identifier -> reference to previously declared space
        return TypeName(name=tok.value, line=tok.line, col=tok.col)

    def _parse_space_args(self) -> tuple[list[str], dict[str, str]]:
        """Parse constructor arguments: positional and keyword.

        Returns
        -------
        tuple[list[str], dict[str, str]]
            Positional args and keyword args.
        """
        args: list[str] = []
        kwargs: dict[str, str] = {}

        if self._peek_type() == TokenType.RPAREN:
            return args, kwargs

        # parse first argument
        self._parse_one_space_arg(args, kwargs)

        while self._peek_type() == TokenType.COMMA:
            self._advance()
            self._parse_one_space_arg(args, kwargs)

        return args, kwargs

    def _parse_one_space_arg(
        self, args: list[str], kwargs: dict[str, str],
    ) -> None:
        """Parse a single positional or keyword argument."""
        tok = self._current()

        # keyword argument: IDENT = value
        if (
            tok.type == TokenType.IDENT
            and self._peek_ahead() == TokenType.EQUALS
        ):
            key_tok = self._advance()
            self._advance()  # consume =
            val_tok = self._advance()
            kwargs[key_tok.value] = val_tok.value
            return

        # positional: INT, FLOAT, or IDENT
        if tok.type in (TokenType.INT, TokenType.FLOAT, TokenType.IDENT):
            self._advance()
            args.append(tok.value)
            return

        raise ParseError("expected argument", tok)

    # -- continuous/stochastic/boundary declarations -------------------------

    def _parse_continuous_decl(self) -> ContinuousMorphismDecl:
        """Parse: continuous <name>[N] : <dom> -> <cod> ~ <family> [opts]."""
        tok = self._expect(TokenType.CONTINUOUS)
        name_tok = self._expect(TokenType.IDENT)

        # optional replication count: name[N]
        replicate = None

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            count_tok = self._expect(TokenType.INT)
            replicate = int(count_tok.value)

            if replicate < 1:
                raise ParseError("replication count must be >= 1", count_tok)

            self._expect(TokenType.RBRACKET)

        self._expect(TokenType.COLON)
        domain_type = self._parse_type_expr()
        self._expect(TokenType.ARROW)
        codomain_type = self._parse_type_expr()
        self._expect(TokenType.TILDE)
        family_tok = self._expect(TokenType.IDENT)

        # optional [key=val, ...] options
        options: dict[str, str] = {}

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            options = self._parse_options()
            self._expect(TokenType.RBRACKET)

        return ContinuousMorphismDecl(
            name=name_tok.value,
            domain=domain_type,
            codomain=codomain_type,
            family=family_tok.value,
            options=options,
            replicate=replicate,
            line=tok.line,
            col=tok.col,
        )

    def _parse_stochastic_decl(self) -> StochasticMorphismDecl:
        """Parse: stochastic <name>[N] : <dom> -> <cod>."""
        tok = self._expect(TokenType.STOCHASTIC)
        name_tok = self._expect(TokenType.IDENT)

        # optional replication count: name[N]
        replicate = None

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            count_tok = self._expect(TokenType.INT)
            replicate = int(count_tok.value)

            if replicate < 1:
                raise ParseError("replication count must be >= 1", count_tok)

            self._expect(TokenType.RBRACKET)

        self._expect(TokenType.COLON)
        domain = self._parse_type_expr()
        self._expect(TokenType.ARROW)
        codomain = self._parse_type_expr()
        return StochasticMorphismDecl(
            name=name_tok.value,
            domain=domain,
            codomain=codomain,
            replicate=replicate,
            line=tok.line,
            col=tok.col,
        )

    def _parse_discretize_decl(self) -> DiscretizeDecl:
        """Parse: discretize <name> : <space> -> <int>."""
        tok = self._expect(TokenType.DISCRETIZE)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.COLON)
        space_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.ARROW)
        bins_tok = self._expect(TokenType.INT)

        options: dict[str, str] = {}

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            options = self._parse_options()
            self._expect(TokenType.RBRACKET)

        return DiscretizeDecl(
            name=name_tok.value,
            space_name=space_tok.value,
            n_bins=int(bins_tok.value),
            options=options,
            line=tok.line,
            col=tok.col,
        )

    def _parse_embed_decl(self) -> EmbedDecl:
        """Parse: embed <name>[N] : <finset> -> <space>."""
        tok = self._expect(TokenType.EMBED)
        name_tok = self._expect(TokenType.IDENT)

        # optional replication count: name[N]
        replicate = None

        if self._peek_type() == TokenType.LBRACKET:
            self._advance()
            count_tok = self._expect(TokenType.INT)
            replicate = int(count_tok.value)

            if replicate < 1:
                raise ParseError("replication count must be >= 1", count_tok)

            self._expect(TokenType.RBRACKET)

        self._expect(TokenType.COLON)
        dom_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.ARROW)
        cod_tok = self._expect(TokenType.IDENT)
        return EmbedDecl(
            name=name_tok.value,
            domain_name=dom_tok.value,
            codomain_name=cod_tok.value,
            replicate=replicate,
            line=tok.line,
            col=tok.col,
        )

    # -- program blocks ------------------------------------------------------

    def _parse_program_decl(self) -> ProgramDecl:
        """Parse a monadic program block.

        ::

            program <name>[(<params>)] : <type_expr> -> <type_expr>
                draw <var_pattern> ~ <morphism>[(<arg_list>)]
                ...
                return <var> | return (<var>, <var>, ...)
        """
        tok = self._expect(TokenType.PROGRAM)
        name_tok = self._expect(TokenType.IDENT)

        # optional named parameters: program name(x, y) : ...
        params = None

        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            param_names = [self._expect(TokenType.IDENT).value]

            while self._peek_type() == TokenType.COMMA:
                self._advance()
                param_names.append(self._expect(TokenType.IDENT).value)

            self._expect(TokenType.RPAREN)
            params = tuple(param_names)

        self._expect(TokenType.COLON)
        domain = self._parse_type_expr()
        self._expect(TokenType.ARROW)
        codomain = self._parse_type_expr()

        # parse draw, observe, and let steps until we hit return
        draws: list[DrawStep | LetStep] = []
        self._skip_newlines()

        while self._peek_type() in (TokenType.DRAW, TokenType.OBSERVE, TokenType.LET) or (
            self._peek_type() == TokenType.IDENT and self._peek_ahead() == TokenType.LARROW
        ):
            if self._peek_type() == TokenType.DRAW:
                draws.append(self._parse_draw_step())

            elif self._peek_type() == TokenType.OBSERVE:
                draws.append(self._parse_observe_step())

            elif self._peek_type() == TokenType.LET:
                draws.append(self._parse_let_step())

            else:
                draws.append(self._parse_arrow_draw_step())

            self._skip_newlines()

        if not draws:
            raise ParseError("program block requires at least one step", self._current())

        # parse return statement: return <var> | return (<var>, ...)
        # optionally with labels: return (label: var, ...)
        self._expect(TokenType.RETURN)
        return_vars, return_labels = self._parse_return_pattern()

        return ProgramDecl(
            name=name_tok.value,
            params=params,
            domain=domain,
            codomain=codomain,
            draws=tuple(draws),
            return_vars=return_vars,
            return_labels=return_labels,
            line=tok.line,
            col=tok.col,
        )

    def _parse_return_pattern(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
        """Parse a return pattern, optionally with labels.

        Supports::

            return x              -> (("x",), None)
            return (x, y)         -> (("x", "y"), None)
            return (a: x, b: y)   -> (("x", "y"), ("a", "b"))

        Returns
        -------
        tuple[tuple[str, ...], tuple[str, ...] | None]
            Variable names and optional labels.
        """
        if self._peek_type() != TokenType.LPAREN:
            # simple single-variable return
            return (self._expect(TokenType.IDENT).value,), None

        self._advance()  # consume LPAREN
        first_name = self._expect(TokenType.IDENT)

        # check if labeled: IDENT COLON IDENT
        if self._peek_type() == TokenType.COLON:
            # labeled return
            self._advance()  # consume COLON
            first_var = self._expect(TokenType.IDENT)
            labels = [first_name.value]
            vars_list = [first_var.value]

            while self._peek_type() == TokenType.COMMA:
                self._advance()

                # trailing comma
                if self._peek_type() == TokenType.RPAREN:
                    break

                label = self._expect(TokenType.IDENT)
                self._expect(TokenType.COLON)
                var = self._expect(TokenType.IDENT)
                labels.append(label.value)
                vars_list.append(var.value)

            self._expect(TokenType.RPAREN)
            return tuple(vars_list), tuple(labels)

        # unlabeled tuple return: (var, var, ...)
        vars_list = [first_name.value]

        while self._peek_type() == TokenType.COMMA:
            self._advance()

            # trailing comma
            if self._peek_type() == TokenType.RPAREN:
                break

            vars_list.append(self._expect(TokenType.IDENT).value)

        self._expect(TokenType.RPAREN)
        return tuple(vars_list), None

    def _parse_draw_step(self) -> DrawStep:
        """Parse: draw <var_pattern> ~ <morphism> ['(' <arg_list> ')'].

        var_pattern is either a single IDENT or ``(IDENT, IDENT, ...)``.
        arg_list is one or more comma-separated items, each of which
        can be an IDENT (variable reference) or INT/FLOAT (literal).
        """
        tok = self._expect(TokenType.DRAW)

        # parse binding pattern: x or (x, y, ...)
        var_pattern = self._parse_var_pattern()

        self._expect(TokenType.TILDE)
        morphism_tok = self._expect(TokenType.IDENT)

        # optional arguments: (arg) or (arg1, arg2, ...)
        args = None

        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            arg_list: list[str | float] = [self._parse_draw_arg()]

            while self._peek_type() == TokenType.COMMA:
                self._advance()
                arg_list.append(self._parse_draw_arg())

            self._expect(TokenType.RPAREN)
            args = tuple(arg_list)

        return DrawStep(
            vars=var_pattern,
            morphism=morphism_tok.value,
            args=args,
            line=tok.line,
            col=tok.col,
        )

    def _parse_observe_step(self) -> DrawStep:
        """Parse: observe <var_pattern> ~ <morphism> ['(' <arg_list> ')'].

        Identical to draw_step but sets is_observed=True on the
        resulting DrawStep node.
        """
        tok = self._expect(TokenType.OBSERVE)

        # parse binding pattern: x or (x, y, ...)
        var_pattern = self._parse_var_pattern()

        self._expect(TokenType.TILDE)
        morphism_tok = self._expect(TokenType.IDENT)

        # optional arguments: (arg) or (arg1, arg2, ...)
        args = None

        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            arg_list: list[str | float] = [self._parse_draw_arg()]

            while self._peek_type() == TokenType.COMMA:
                self._advance()
                arg_list.append(self._parse_draw_arg())

            self._expect(TokenType.RPAREN)
            args = tuple(arg_list)

        return DrawStep(
            vars=var_pattern,
            morphism=morphism_tok.value,
            args=args,
            is_observed=True,
            line=tok.line,
            col=tok.col,
        )

    def _parse_arrow_draw_step(self) -> DrawStep:
        """Parse: IDENT '<-' IDENT ['(' arg_list ')'].

        Alternative do-notation style bind syntax.
        """
        var_tok = self._expect(TokenType.IDENT)
        tok = self._expect(TokenType.LARROW)
        morphism_tok = self._expect(TokenType.IDENT)

        args = None
        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            arg_list: list[str | float] = [self._parse_draw_arg()]
            while self._peek_type() == TokenType.COMMA:
                self._advance()
                arg_list.append(self._parse_draw_arg())
            self._expect(TokenType.RPAREN)
            args = tuple(arg_list)

        return DrawStep(
            vars=(var_tok.value,),
            morphism=morphism_tok.value,
            args=args,
            line=tok.line,
            col=tok.col,
        )

    def _parse_draw_arg(self) -> str | float:
        """Parse a single draw argument: IDENT or number literal.

        Supports negative literals via a leading minus sign.

        Returns
        -------
        str or float
            Variable name (str) or literal value (float).
        """
        tok = self._current()

        # handle negative numeric literals: -1.0, -3, etc.
        if tok.type == TokenType.MINUS:
            self._advance()
            num_tok = self._current()

            if num_tok.type in (TokenType.FLOAT, TokenType.INT):
                self._advance()
                return -float(num_tok.value)

            raise ParseError("expected number after '-'", num_tok)

        if tok.type == TokenType.IDENT:
            self._advance()
            return tok.value

        elif tok.type == TokenType.FLOAT:
            self._advance()
            return float(tok.value)

        elif tok.type == TokenType.INT:
            self._advance()
            return float(tok.value)

        else:
            raise ParseError("expected variable name or number literal", tok)

    def _parse_let_step(self) -> LetStep:
        """Parse: let IDENT = let_expr.

        Binds a variable to a constant literal, variable alias, or
        an arithmetic expression over bound variables and built-in
        functions.
        """
        tok = self._expect(TokenType.LET)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.EQUALS)

        # try simple cases first: bare IDENT or bare number
        # (with no following operator)
        val_tok = self._current()
        next_type = self._peek_ahead()

        # simple variable alias: IDENT not followed by operator or (
        if (
            val_tok.type == TokenType.IDENT
            and val_tok.value not in _LET_BUILTINS
            and next_type not in (
                TokenType.PRODUCT, TokenType.COPRODUCT,
                TokenType.LPAREN,
            )
            # check for + and - via peek at actual token values
            and not self._is_let_operator(next_type)
        ):
            self._advance()
            return LetStep(
                name=name_tok.value,
                value=val_tok.value,
                line=tok.line,
                col=tok.col,
            )

        # simple numeric literal not followed by operator
        if (
            val_tok.type in (TokenType.FLOAT, TokenType.INT)
            and not self._is_let_operator(next_type)
        ):
            self._advance()
            return LetStep(
                name=name_tok.value,
                value=float(val_tok.value),
                line=tok.line,
                col=tok.col,
            )

        # full expression parse
        expr_node = self._parse_let_expr()

        return LetStep(
            name=name_tok.value,
            value=expr_node,
            line=tok.line,
            col=tok.col,
        )

    def _is_let_operator(self, ttype: TokenType) -> bool:
        """Check if a token type is an arithmetic operator for let exprs."""
        return ttype in (
            TokenType.COPRODUCT,  # +
            TokenType.PRODUCT,    # *
            TokenType.MINUS,      # -
            TokenType.SLASH,      # /
        )

    # -- let expression parser (arithmetic) ----------------------------------

    def _parse_let_expr(self) -> LetExprNode:
        """Parse a let expression: additive level (lowest precedence).

        Grammar::

            let_expr  := let_term (('+' | '-') let_term)*
            let_term  := let_unary (('*' | '/') let_unary)*
            let_unary := '-' let_atom | let_atom
            let_atom  := FLOAT | INT | IDENT '(' let_expr (',' let_expr)* ')'
                       | IDENT | '(' let_expr ')'
        """
        left = self._parse_let_term()

        while True:
            tok = self._current()

            if tok.type == TokenType.COPRODUCT:
                # + operator
                self._advance()
                right = self._parse_let_term()
                left = LetExprBinOp("+", left, right)

            elif tok.type == TokenType.MINUS:
                # - operator (subtraction)
                self._advance()
                right = self._parse_let_term()
                left = LetExprBinOp("-", left, right)

            else:
                break

        return left

    def _parse_let_term(self) -> LetExprNode:
        """Parse multiplicative level: let_unary (('*' | '/') let_unary)*."""
        left = self._parse_let_unary()

        while self._peek_type() in (TokenType.PRODUCT, TokenType.SLASH):
            tok = self._current()
            op = "*" if tok.type == TokenType.PRODUCT else "/"
            self._advance()
            right = self._parse_let_unary()
            left = LetExprBinOp(op, left, right)

        return left

    def _parse_let_unary(self) -> LetExprNode:
        """Parse unary: '-' let_atom | let_atom."""
        if self._peek_type() == TokenType.MINUS:
            self._advance()
            operand = self._parse_let_atom()
            return LetExprUnaryOp(operand)

        return self._parse_let_atom()

    def _parse_let_atom(self) -> LetExprNode:
        """Parse atomic let expression."""
        tok = self._current()

        if tok.type in (TokenType.FLOAT, TokenType.INT):
            self._advance()
            return LetExprLiteral(float(tok.value))

        if tok.type == TokenType.IDENT:
            # check if it's a builtin function call
            if tok.value in _LET_BUILTINS and self._peek_ahead() == TokenType.LPAREN:
                func_name = tok.value
                self._advance()  # consume function name
                self._advance()  # consume (

                args = [self._parse_let_expr()]

                while self._peek_type() == TokenType.COMMA:
                    self._advance()
                    args.append(self._parse_let_expr())

                self._expect(TokenType.RPAREN)
                return LetExprCall(func_name, tuple(args))

            # bare variable reference
            self._advance()
            return LetExprVar(tok.value)

        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._parse_let_expr()
            self._expect(TokenType.RPAREN)
            return node

        raise ParseError(
            "expected number, variable, or expression in let binding", tok,
        )

    def _parse_var_pattern(self) -> tuple[str, ...]:
        """Parse a variable pattern: IDENT or '(' IDENT (',' IDENT)* ')'.

        Returns
        -------
        tuple[str, ...]
            Single-element tuple for simple binding, multi-element
            for destructuring.
        """
        if self._peek_type() == TokenType.LPAREN:
            self._advance()
            names = [self._expect(TokenType.IDENT).value]

            while self._peek_type() == TokenType.COMMA:
                self._advance()

                # allow trailing comma: (a,)
                if self._peek_type() == TokenType.RPAREN:
                    break

                names.append(self._expect(TokenType.IDENT).value)

            self._expect(TokenType.RPAREN)
            return tuple(names)

        return (self._expect(TokenType.IDENT).value,)

    # -- type expressions ----------------------------------------------------

    def _parse_type_expr(self) -> TypeExpr:
        """Parse a type expression (coproduct is lowest precedence)."""
        return self._parse_coproduct_type()

    def _parse_coproduct_type(self) -> TypeExpr:
        """Parse: product_type ('+' product_type)*."""
        left = self._parse_product_type()
        components = [left]

        while self._peek_type() == TokenType.COPRODUCT:
            self._advance()
            components.append(self._parse_product_type())

        if len(components) == 1:
            return components[0]

        return TypeCoproduct(
            components=tuple(components),
            line=components[0].line if hasattr(components[0], "line") else 0,
            col=components[0].col if hasattr(components[0], "col") else 0,
        )

    def _parse_product_type(self) -> TypeExpr:
        """Parse: primary_type ('*' primary_type)*."""
        left = self._parse_primary_type()
        components = [left]

        while self._peek_type() == TokenType.PRODUCT:
            self._advance()
            components.append(self._parse_primary_type())

        if len(components) == 1:
            return components[0]

        return TypeProduct(
            components=tuple(components),
            line=components[0].line if hasattr(components[0], "line") else 0,
            col=components[0].col if hasattr(components[0], "col") else 0,
        )

    def _parse_primary_type(self) -> TypeExpr:
        """Parse: IDENT | INT | '(' type_expr ')'."""
        tok = self._current()

        if tok.type == TokenType.IDENT:
            self._advance()
            return TypeName(name=tok.value, line=tok.line, col=tok.col)

        elif tok.type == TokenType.INT:
            self._advance()
            return TypeName(name=tok.value, line=tok.line, col=tok.col)

        elif tok.type == TokenType.LPAREN:
            self._advance()
            texpr = self._parse_type_expr()
            self._expect(TokenType.RPAREN)
            return texpr

        else:
            raise ParseError("expected type expression", tok)

    # -- value expressions ---------------------------------------------------

    def _parse_expr(self) -> Expr:
        """Parse a value expression (top level)."""
        return self._parse_compose_expr()

    def _parse_compose_expr(self) -> Expr:
        """Parse: tensor_expr ('>>' | '<<' | '>=' tensor_expr)*."""
        left = self._parse_tensor_expr()

        while self._peek_type() in (TokenType.COMPOSE, TokenType.COMPOSE_BACK, TokenType.KLEISLI):
            tok = self._advance()
            right = self._parse_tensor_expr()

            if tok.type == TokenType.COMPOSE_BACK:
                # backward: f << g means g >> f
                left = ExprCompose(
                    left=right, right=left, line=tok.line, col=tok.col
                )
            else:
                # forward: f >> g or f >=> g (both same semantics)
                left = ExprCompose(
                    left=left, right=right, line=tok.line, col=tok.col
                )

        return left

    def _parse_tensor_expr(self) -> Expr:
        """Parse: postfix_expr ('@' postfix_expr)*."""
        left = self._parse_postfix_expr()

        while self._peek_type() == TokenType.TENSOR:
            tok = self._advance()
            right = self._parse_postfix_expr()
            left = ExprTensorProduct(
                left=left, right=right, line=tok.line, col=tok.col
            )

        return left

    def _parse_postfix_expr(self) -> Expr:
        """Parse: atom_expr ('.' method_call)*."""
        expr = self._parse_atom_expr()

        while self._peek_type() == TokenType.DOT:
            self._advance()
            tok = self._current()

            if tok.type == TokenType.MARGINALIZE:
                self._advance()
                self._expect(TokenType.LPAREN)
                names = [self._expect(TokenType.IDENT).value]

                while self._peek_type() == TokenType.COMMA:
                    self._advance()
                    names.append(self._expect(TokenType.IDENT).value)

                self._expect(TokenType.RPAREN)
                expr = ExprMarginalize(
                    inner=expr,
                    names=tuple(names),
                    line=tok.line,
                    col=tok.col,
                )

            else:
                raise ParseError("expected method name", tok)

        return expr

    def _parse_atom_expr(self) -> Expr:
        """Parse: 'identity' '(' IDENT ')' | 'fan' '(' expr_list ')'
                | 'repeat' '(' expr ',' INT ')' | 'stack' '(' expr ',' INT ')'
                | IDENT | '(' expr ')'."""
        tok = self._current()

        if tok.type == TokenType.IDENTITY:
            self._advance()
            self._expect(TokenType.LPAREN)
            name_tok = self._expect(TokenType.IDENT)
            self._expect(TokenType.RPAREN)
            return ExprIdentity(
                object_name=name_tok.value, line=tok.line, col=tok.col
            )

        elif tok.type == TokenType.FAN:
            self._advance()
            self._expect(TokenType.LPAREN)
            exprs = [self._parse_expr()]

            while self._peek_type() == TokenType.COMMA:
                self._advance()
                exprs.append(self._parse_expr())

            self._expect(TokenType.RPAREN)
            return ExprFan(
                exprs=tuple(exprs), line=tok.line, col=tok.col,
            )

        elif tok.type == TokenType.REPEAT:
            self._advance()
            self._expect(TokenType.LPAREN)
            inner = self._parse_expr()

            # count is optional: repeat(f, 3) or repeat(f)
            count: int | None = None

            if self._peek_type() == TokenType.COMMA:
                self._advance()
                count_tok = self._expect(TokenType.INT)
                count = int(count_tok.value)

                if count < 1:
                    raise ParseError("repeat count must be >= 1", count_tok)

            self._expect(TokenType.RPAREN)
            return ExprRepeat(
                expr=inner, count=count, line=tok.line, col=tok.col,
            )

        elif tok.type == TokenType.STACK:
            self._advance()
            self._expect(TokenType.LPAREN)
            inner = self._parse_expr()
            self._expect(TokenType.COMMA)
            count_tok = self._expect(TokenType.INT)
            count = int(count_tok.value)

            if count < 1:
                raise ParseError("stack count must be >= 1", count_tok)

            self._expect(TokenType.RPAREN)
            return ExprStack(
                expr=inner, count=count, line=tok.line, col=tok.col,
            )

        elif tok.type == TokenType.SCAN:
            self._advance()
            self._expect(TokenType.LPAREN)
            inner = self._parse_expr()

            # optional init parameter: scan(cell, init=learned)
            init = "zeros"

            if self._peek_type() == TokenType.COMMA:
                self._advance()
                init_key = self._expect(TokenType.IDENT)

                if init_key.value != "init":
                    raise ParseError(
                        f"expected 'init', got {init_key.value!r}",
                        init_key,
                    )

                self._expect(TokenType.EQUALS)
                init_val = self._expect(TokenType.IDENT)

                if init_val.value not in ("zeros", "learned"):
                    raise ParseError(
                        f"unknown init strategy {init_val.value!r}; "
                        f"expected 'zeros' or 'learned'",
                        init_val,
                    )

                init = init_val.value

            self._expect(TokenType.RPAREN)
            return ExprScan(
                expr=inner, init=init, line=tok.line, col=tok.col,
            )

        elif tok.type == TokenType.PARSER:
            # parser(categories=[...], rules=[...], start=S)
            return self._parse_parser_expr(tok, default_rules=None)

        elif tok.type == TokenType.CCG:
            # ccg(...) is sugar for parser(...) with CCG default rules
            return self._parse_parser_expr(tok, default_rules=(
                "evaluation", "harmonic_composition", "crossed_composition",
            ))

        elif tok.type == TokenType.LAMBEK:
            # lambek(...) is sugar for parser(...) with Lambek default rules
            return self._parse_parser_expr(tok, default_rules=(
                "evaluation", "adjunction_units",
                "tensor_introduction", "tensor_projection",
            ))

        elif tok.type == TokenType.IDENT:
            self._advance()
            return ExprIdent(name=tok.value, line=tok.line, col=tok.col)

        elif tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        else:
            raise ParseError("expected expression", tok)

    # ------------------------------------------------------------------
    # parser / ccg / lambek shared parser
    # ------------------------------------------------------------------

    def _parse_parser_expr(
        self,
        tok: Token,
        default_rules: tuple[str, ...] | None,
    ) -> ExprParser:
        """Parse the ``parser()`` keyword.

        Syntax::

            parser(
                rules=[...],
                categories=[...],   # optional
                terminal=Token,     # required for schema rules
                start=S,            # optional
                depth=1,            # optional
                constructors=[...]  # optional
            )

        Every entry in ``rules`` is an identifier resolved at compile
        time — either as a schema name in ``SCHEMA_REGISTRY`` or as a
        declared morphism whose type determines its deductive role.

        For ``ccg`` and ``lambek`` aliases, ``default_rules`` provides
        the standard rule set and ``rules=[...]`` is optional.

        Parameters
        ----------
        tok : Token
            The keyword token (PARSER, CCG, or LAMBEK).
        default_rules : tuple of str or None
            Default rule names when rules= is omitted.
        """
        self._advance()
        self._expect(TokenType.LPAREN)

        categories: list[str] = []
        rules: list[str] | None = None
        terminal: str | None = None
        start: str | int = "S"
        depth = 1
        constructors: list[str] | None = None

        # parse keyword arguments in any order
        while self._peek_type() == TokenType.IDENT:
            key_tok = self._expect(TokenType.IDENT)
            key = key_tok.value

            if key == "categories":
                self._expect(TokenType.EQUALS)
                categories = self._parse_ident_list()

            elif key == "rules":
                self._expect(TokenType.EQUALS)
                rules = self._parse_ident_list()

            elif key == "terminal":
                self._expect(TokenType.EQUALS)
                term_tok = self._expect(TokenType.IDENT)
                terminal = term_tok.value

            elif key == "start":
                self._expect(TokenType.EQUALS)
                start_tok = self._current()

                if start_tok.type == TokenType.INT:
                    self._advance()
                    start = int(start_tok.value)

                elif start_tok.type == TokenType.IDENT:
                    self._advance()
                    start = start_tok.value

                else:
                    raise ParseError(
                        "expected category name or integer index "
                        "for start=",
                        start_tok,
                    )

            elif key == "depth":
                self._expect(TokenType.EQUALS)
                depth_tok = self._expect(TokenType.INT)
                depth = int(depth_tok.value)

            elif key == "constructors":
                self._expect(TokenType.EQUALS)
                constructors = self._parse_ident_list()

            else:
                raise ParseError(
                    f"unexpected keyword argument {key!r}; "
                    f"expected 'rules', 'categories', 'terminal', "
                    f"'start', 'depth', or 'constructors'",
                    key_tok,
                )

            # consume trailing comma if present
            if self._peek_type() == TokenType.COMMA:
                self._advance()

        self._expect(TokenType.RPAREN)

        # resolve rules: explicit > default > error
        if rules is not None:
            rule_tuple = tuple(rules)

        elif default_rules is not None:
            rule_tuple = default_rules

        else:
            raise ParseError(
                "parser() requires rules=[...]",
                tok,
            )

        return ExprParser(
            rules=rule_tuple,
            categories=tuple(categories),
            terminal=terminal,
            start=start,
            depth=depth,
            constructors=(
                tuple(constructors) if constructors is not None
                else None
            ),
            line=tok.line,
            col=tok.col,
        )

    def _parse_ident_list(self) -> list[str]:
        """Parse a bracketed list of identifiers: [A, B, C]."""
        self._expect(TokenType.LBRACKET)

        items: list[str] = []
        item_tok = self._expect(TokenType.IDENT)
        items.append(item_tok.value)

        while self._peek_type() == TokenType.COMMA:
            self._advance()

            if self._peek_type() == TokenType.RBRACKET:
                break

            item_tok = self._expect(TokenType.IDENT)
            items.append(item_tok.value)

        self._expect(TokenType.RBRACKET)
        return items
