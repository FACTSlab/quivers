/**
 * @file Quivers DSL grammar
 * @author Aaron Steven White <aaronstevenwhite@gmail.com>
 * @license MIT
 *
 * QVR is a domain-specific language for declaring categorical theories:
 * quantales, objects, morphisms, continuous and stochastic spaces, and
 * monadic programs over them. Each .qvr file defines a schema in the
 * sense of panproto: vertices for quantales/objects/spaces, edges for
 * morphisms and dependencies, and a program over that schema.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  compose: 1,    // >> << >=>
  tensor:  2,    // @
  postfix: 3,    // .method(...)
  // type expression precedence:
  type_coproduct: 1,  // +
  type_slash:     2,  // / \   (residuated; binds tighter than +, looser than *)
  type_product:   3,  // *
  type_apply:     4,  // T(X)  effect-typed application
  // let-arithmetic precedence:
  let_add: 1,
  let_mul: 2,
  let_unary: 3,
};

module.exports = grammar({
  name: 'qvr',

  extras: $ => [/\s/, $.doc_comment, $.line_comment],

  word: $ => $.identifier,

  conflicts: $ => [],

  rules: {
    // ---------------------------------------------------------------
    // top level
    // ---------------------------------------------------------------
    source_file: $ => repeat($._statement),

    _statement: $ => choice(
      $.quantale_decl,
      $.category_decl,
      $.rule_decl,
      $.schema_decl,
      $.object_decl,
      $.morphism_decl,
      $.space_decl,
      $.type_alias_decl,
      $.alias_decl,
      $.bundle_decl,
      $.continuous_decl,
      $.stochastic_decl,
      $.discretize_decl,
      $.embed_decl,
      $.program_decl,
      $.let_decl,
      $.export_decl,
    ),

    // ---------------------------------------------------------------
    // simple declarations
    // ---------------------------------------------------------------

    quantale_decl: $ => seq('quantale', field('name', $.identifier)),

    category_decl: $ => seq(
      'category',
      field('names', commaSep1($.identifier)),
    ),

    // `object X : 3` — anonymous-element FinSet of cardinality 3 (or
    // a TypeExpr binding for products / coproducts).
    // `object Atoms = {NP, S, VP}` — named-element EnumSet.
    // `object Cat = FreeResiduated(Atoms, depth=4, ops=[/, \\, *])`
    //               — residuated category universe over an EnumSet.
    object_decl: $ => seq(
      'object',
      field('name', $.identifier),
      choice(
        seq(':', field('type', $._type_expr)),
        seq('=', field('init', $._object_initializer)),
      ),
    ),

    _object_initializer: $ => choice(
      $.enum_set_literal,
      $.free_residuated_expr,
      $.free_monoid_expr,
    ),

    enum_set_literal: $ => seq(
      '{',
      field('elements', commaSep1($.identifier)),
      '}',
    ),

    free_residuated_expr: $ => seq(
      'FreeResiduated',
      '(',
      field('generators', $.identifier),
      optional(seq(
        ',',
        commaSep1($.free_residuated_arg),
      )),
      ')',
    ),

    free_residuated_arg: $ => choice(
      seq('depth', '=', field('depth', $.integer)),
      seq('ops', '=', '[', commaSep1(field('op', $.identifier)), ']'),
    ),

    // FreeMonoid(generators, max_length=N) — bounded Kleene closure.
    free_monoid_expr: $ => seq(
      'FreeMonoid',
      '(',
      field('generators', $.identifier),
      ',',
      'max_length', '=', field('max_length', $.integer),
      ')',
    ),

    morphism_decl: $ => seq(
      field('kind', choice('latent', 'observed')),
      field('name', $.identifier),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      optional(field('options', $.option_block)),
      optional(seq('=', field('init', $._expr))),
    ),

    let_decl: $ => prec.right(seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._expr),
      optional(seq(
        'where',
        field('where', repeat1($.let_decl)),
      )),
    )),

    // Module-level export: `export E`. Any number per module; each
    // selects a top-level morphism / posterior / deduction for the
    // compiled output. Replaces the v0.4 `output` keyword (which
    // permitted exactly one) — semantically a public binding.
    export_decl: $ => seq('export', field('value', $._expr)),

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    // rule declarations (CCG/Lambek-style)
    // ---------------------------------------------------------------

    rule_decl: $ => seq(
      'rule',
      field('name', $.identifier),
      '(',
      field('variables', commaSep1($.identifier)),
      ')',
      ':',
      field('premises', commaSep1($._type_expr)),
      '=>',
      field('conclusion', $._type_expr),
    ),

    // `schema r[X, Y : Cat] : (X/Y) * Y -> X` — pattern-polymorphic
    // morphism schema. Domain shape determines arity: a 2-component
    // product domain produces a binary chart-rule; a single-component
    // domain produces a unary rule.
    schema_decl: $ => seq(
      'schema',
      field('name', $.identifier),
      '[',
      field('parameters', commaSep1($.schema_parameter)),
      ']',
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
    ),

    schema_parameter: $ => seq(
      field('names', commaSep1($.identifier)),
      ':',
      field('type', $._type_expr),
    ),

    // ---------------------------------------------------------------
    // type expressions  (categorical objects: products and coproducts of finsets)
    // ---------------------------------------------------------------

    _type_expr: $ => choice(
      $.type_coproduct,
      $.type_slash,
      $.type_product,
      $.type_effect_apply,
      $.type_atom,
      $.type_paren,
    ),

    type_atom: $ => choice($.identifier, $.integer),

    type_paren: $ => seq('(', $._type_expr, ')'),

    type_product: $ => prec.left(PREC.type_product, seq(
      field('left',  $._type_expr),
      '*',
      field('right', $._type_expr),
    )),

    type_coproduct: $ => prec.left(PREC.type_coproduct, seq(
      field('left',  $._type_expr),
      '+',
      field('right', $._type_expr),
    )),

    type_slash: $ => prec.left(PREC.type_slash, seq(
      field('result',    $._type_expr),
      field('direction', choice('/', '\\')),
      field('argument',  $._type_expr),
    )),

    // T(X)  — effect-typed application.
    // The named effect must already be a fully-instantiated effect
    // (parameters baked into its declared name; e.g. `Cont_S(NP)`,
    // not `Cont[S](NP)`). This avoids parse ambiguity with the
    // `[option_block]` that may follow a morphism's codomain.
    type_effect_apply: $ => prec(PREC.type_apply, seq(
      field('effect', $.identifier),
      '(',
      field('args', commaSep1($._type_expr)),
      ')',
    )),

    // ---------------------------------------------------------------
    // space expressions  (continuous spaces)
    // ---------------------------------------------------------------

    space_decl: $ => seq(
      'space',
      field('name', $.identifier),
      ':',
      field('value', $._space_expr),
    ),

    // ML-style: `type Latent = Euclidean 16`
    // ML-style: `type Latent = Euclidean 16`
    type_alias_decl: $ => seq(
      'type',
      field('name', $.identifier),
      '=',
      field('value', $._space_expr),
    ),

    // `alias Foo = X * Y` — object-level type alias. Distinct keyword
    // from `type` to keep the parse unambiguous between the
    // overlapping type_atom and space_atom productions.
    alias_decl: $ => seq(
      'alias',
      field('name', $.identifier),
      '=',
      field('value', $._type_expr),
    ),

    // `bundle CCG = [forward_app, backward_app]` — first-class
    // schema-bundle binding. parser(rules=CCG) and chart_fold's
    // schema-set arguments accept the bundle by name.
    bundle_decl: $ => seq(
      'bundle',
      field('name', $.identifier),
      '=',
      '[',
      optional(field('rules', commaSep1($.identifier))),
      ']',
    ),

    _space_expr: $ => choice(
      $.space_product,
      $.space_constructor,
      $.space_constructor_bare,
      $.space_atom,
    ),

    space_atom: $ => $.identifier,

    space_product: $ => prec.left(PREC.type_product, seq(
      field('left',  $._space_expr),
      '*',
      field('right', $._space_expr),
    )),

    // parenthesized: `Euclidean(16)`, `Euclidean(2, low=0.0, high=1.0)`
    space_constructor: $ => seq(
      field('constructor', $.identifier),
      '(',
      optional(field('args', commaSep1($._space_arg))),
      ')',
    ),

    // bareword form: `Euclidean 16` (one numeric positional arg, no parens)
    space_constructor_bare: $ => prec(1, seq(
      field('constructor', $.identifier),
      field('arg', $._numeric_literal),
    )),

    _space_arg: $ => choice(
      $.space_kwarg,
      $.integer,
      $.float,
    ),

    space_kwarg: $ => seq(
      field('key', $.identifier),
      '=',
      field('value', $._numeric_literal),
    ),

    _numeric_literal: $ => choice($.integer, $.float),

    // ---------------------------------------------------------------
    // continuous / stochastic / discretize / embed declarations
    // ---------------------------------------------------------------

    continuous_decl: $ => seq(
      'continuous',
      field('name', $.identifier),
      optional(field('replicate', $.replicate_count)),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      '~',
      field('family', $.identifier),
      optional(field('options', $.option_block)),
    ),

    stochastic_decl: $ => seq(
      'stochastic',
      field('name', $.identifier),
      optional(field('replicate', $.replicate_count)),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
    ),

    discretize_decl: $ => seq(
      'discretize',
      field('name', $.identifier),
      ':',
      field('space', $.identifier),
      '->',
      field('bins', $.integer),
      optional(field('options', $.option_block)),
    ),

    embed_decl: $ => seq(
      'embed',
      field('name', $.identifier),
      optional(field('replicate', $.replicate_count)),
      ':',
      field('domain', $.identifier),
      '->',
      field('codomain', $.identifier),
    ),

    replicate_count: $ => seq('[', $.integer, ']'),

    option_block: $ => seq('[', commaSep1($.option_entry), ']'),

    option_entry: $ => seq(
      field('key', $.identifier),
      '=',
      field('value', choice($.identifier, $.integer, $.float)),
    ),

    // ---------------------------------------------------------------
    // value (morphism) expressions
    // ---------------------------------------------------------------

    _expr: $ => choice(
      $.compose_expr,
      $.tensor_expr,
      $.postfix_expr,
      $._atom_expr,
    ),

    compose_expr: $ => prec.left(PREC.compose, seq(
      field('left',  $._expr),
      field('op',    choice('>>', '<<', '>=>')),
      field('right', $._expr),
    )),

    tensor_expr: $ => prec.left(PREC.tensor, seq(
      field('left',  $._expr),
      '@',
      field('right', $._expr),
    )),

    postfix_expr: $ => prec.left(PREC.postfix, seq(
      field('inner', $._expr),
      '.',
      field('method', $.method_call),
    )),

    method_call: $ => choice(
      seq(
        field('name', 'marginalize'),
        '(',
        field('args', commaSep1($.identifier)),
        ')',
      ),
      // residuation-witness combinators (Phase 4): given f : X * Y -> Z
      // where Z lives in a residuated universe, produce f.curry_right :
      // X -> Z/Y or f.curry_left : Y -> X\Z. No arguments.
      seq(field('name', choice('curry_right', 'curry_left'))),
    ),

    _atom_expr: $ => choice(
      $.expr_paren,
      $.identity_expr,
      $.fan_expr,
      $.repeat_expr,
      $.stack_expr,
      $.scan_expr,
      $.parser_expr,
      $.chart_fold_expr,
      $.expr_ident,
    ),

    expr_paren: $ => seq('(', $._expr, ')'),

    expr_ident: $ => $.identifier,

    identity_expr: $ => seq(
      'identity', '(', field('object', $.identifier), ')',
    ),

    fan_expr: $ => seq(
      'fan', '(', field('args', commaSep1($._expr)), ')',
    ),

    // repeat(f) or repeat(f, N)
    repeat_expr: $ => seq(
      'repeat', '(',
      field('inner', $._expr),
      optional(seq(',', field('count', $.integer))),
      ')',
    ),

    stack_expr: $ => seq(
      'stack', '(',
      field('inner', $._expr),
      ',',
      field('count', $.integer),
      ')',
    ),

    scan_expr: $ => seq(
      'scan', '(',
      field('inner', $._expr),
      optional(seq(',', 'init', '=', field('init', $.identifier))),
      ')',
    ),

    parser_expr: $ => seq(
      field('keyword', choice('parser', 'ccg', 'lambek')),
      '(',
      optional(field('args', commaSep1($.parser_arg))),
      ')',
    ),

    // chart_fold(lex=, binary=, unary=, start=, depth=, effect_depth=)
    // — desugared parser-construction primitive of Phase 5. Each
    // keyword argument carries a value that is itself an expression
    // (lex, binary, unary) or an identifier/integer literal
    // (start, depth, effect_depth).
    chart_fold_expr: $ => seq(
      'chart_fold',
      '(',
      optional(field('args', commaSep1($.chart_fold_arg))),
      ')',
    ),

    chart_fold_arg: $ => prec(10, seq(
      field('key', choice(
        'lex', 'binary', 'unary', 'start', 'depth', 'effect_depth',
      )),
      '=',
      field('value', choice(
        $._expr,
        $.integer,
      )),
    )),

    parser_arg: $ => seq(
      field('key', choice(
        'rules', 'categories', 'terminal', 'start', 'depth', 'constructors',
      )),
      '=',
      field('value', choice(
        $.ident_list,
        $.identifier,
        $.integer,
      )),
    ),

    ident_list: $ => seq('[', optional(commaSep1($.identifier)), ']'),

    // ---------------------------------------------------------------
    // program blocks
    // ---------------------------------------------------------------

    // A program is a monadic-kernel block:
    //   program name (params) : dom -> cod ! Sample, Score [over M]
    //       body
    //       return e
    //
    // Parameter list (optional):
    //   - concrete:     bare identifiers `(data1, data2)` naming the
    //                   components of the domain product.
    //   - parametric:   typed `(G : FinSet, s : Real, f : Mor[A,B])`
    //                   denoting a dependent kernel family.
    //
    // Effect signature (optional, after `!`):
    //   A comma-separated list of capabilities the body uses. Empty
    //   set is unannotated; explicit `! Pure` disallows sample/score
    //   steps. Effects: `Sample`, `Score`, `Marginal`, `Pure`.
    //
    // Posterior modifier (optional, after `over`):
    //   `over M` declares this program runs over a per-sample
    //   snapshot of the model M's latent trace — the v0.5 replacement
    //   for the v0.4 `posterior` keyword.
    program_decl: $ => seq(
      'program',
      field('name', $.identifier),
      optional(seq('(', field('params', commaSep1($._program_param)), ')')),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      optional(seq('!', field('effects', commaSep1($.identifier)))),
      optional(seq('over', field('over_model', $.identifier))),
      field('steps', repeat1($._program_step)),
      'return',
      field('return', $._return_pattern),
    ),

    _program_param: $ => choice(
      $.identifier,
      $.typed_program_param,
    ),

    // Typed program parameter: `name : Kind`.
    // Kinds:
    //   FinSet, Space, Object — object-typed (parametric over an
    //     object of the relevant subcategory).
    //   Real, Nat              — scalar-typed (a hyperparameter value).
    //   Mor[Dom, Cod]          — morphism-typed (a kernel of the given
    //                            signature, passed in by name).
    typed_program_param: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $._param_kind),
    ),

    _param_kind: $ => choice(
      $.object_kind,
      $.scalar_kind,
      $.morphism_kind,
    ),

    object_kind: $ => choice('FinSet', 'Space', 'Object'),

    scalar_kind: $ => choice('Real', 'Nat'),

    morphism_kind: $ => seq(
      'Mor',
      '[',
      field('domain', $._type_expr),
      ',',
      field('codomain', $._type_expr),
      ']',
    ),

    _program_step: $ => choice(
      $.marginalize_step,
      $.observe_step,
      $.bind_step,
      $.let_step,
    ),

    // Kleisli bind: the unique sampling-step shape.
    //
    //   v        <- F(args)              -- scalar draw
    //   v : A    <- F(args)              -- A-indexed plate
    //   (a, b)   <- F(args)              -- destructuring tuple bind
    //
    // The optional `: A` annotation declares v as an A-indexed family
    // (categorically a Kern-morphism A → ⟦cod(F)⟧, equivalently a
    // single arrow 1 → ⟦cod(F)⟧^A via the iso Kern(1, K^A) ≅ Kern(A, K)).
    // Arguments may be inline bracket-indexed sections `theta[N]`
    // referring to plate variables.
    bind_step: $ => prec.right(seq(
      field('vars', $._var_pattern),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    )),

    // Scored bind — same shape as `bind_step` but prefixed with
    // `observe`, marking the bound coordinate as clamped at runtime
    // by the `observations` dict; the resulting kernel becomes
    // sub-probabilistic.
    //
    //   observe v        <- F(args)
    //   observe r : N    <- F(theta[N])   -- N-indexed batched score
    observe_step: $ => prec.right(seq(
      'observe',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    )),

    // Scoped marginalisation. Introduces a coordinate `c` bound to a
    // kernel `F(args)`, optionally indexed by `: A`; the body in `{ … }`
    // is the integration scope. At the end of the scope the coordinate
    // is pushed forward through the projection (logsumexp for discrete,
    // fibrewise integration for continuous), and `c` falls out of
    // scope. Replaces v0.4's trailing `marginalize c` form.
    //
    //   marginalize class : Item <- Categorical(probs) in {
    //       observe r : N <- Bernoulli(theta[class[N]])
    //   }
    marginalize_step: $ => seq(
      'marginalize',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
      'in',
      '{',
      field('scope', repeat($._program_step)),
      '}',
    ),

    let_step: $ => seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._let_arith),
    ),

    // A family argument is one of:
    //   - a numeric literal: `1.0`, `-3`
    //   - an identifier: `sigma`, `intercept`
    //   - a bracket-indexed family section: `theta[N]`
    //     where `N` is a type-expr naming a plate's index set.
    // The bracket form annotates that the argument is an N-indexed
    // family — categorically a section of `theta : N → P`.
    _draw_arg: $ => choice(
      $.bracket_index_arg,
      $.identifier,
      $.signed_number,
    ),

    bracket_index_arg: $ => prec(1, seq(
      field('name', $.identifier),
      '[',
      field('index', $._type_expr),
      ']',
    )),

    _var_pattern: $ => choice(
      $.identifier,
      $.var_tuple,
    ),

    var_tuple: $ => seq(
      '(',
      commaSep1($.identifier),
      optional(','),
      ')',
    ),

    _return_pattern: $ => choice(
      $.identifier,
      $.return_tuple,
    ),

    return_tuple: $ => seq(
      '(',
      commaSep1($.identifier),
      optional(','),
      ')',
    ),

    // ---------------------------------------------------------------
    // let-step arithmetic mini-language
    // ---------------------------------------------------------------

    _let_arith: $ => choice(
      $.let_binop,
      $.let_unary,
      $._let_atom,
    ),

    _let_atom: $ => choice(
      $.let_paren,
      $.let_call,
      $.let_index,
      $.let_var,
      $.let_literal,
    ),

    // Indexed access into a finite-domain-indexed family: arr[i, j, ...].
    // Categorically the Kleisli pullback ι^* v = v ∘ ι : N → B for a
    // plate variable v : A → B and a finite fibration ι : N → A.
    let_index: $ => prec.left(seq(
      field('array', $.let_var),
      '[',
      field('indices', commaSep1($._let_arith)),
      ']',
    )),

    let_paren: $ => seq('(', $._let_arith, ')'),

    let_var: $ => $.identifier,

    let_literal: $ => $._numeric_literal,

    let_call: $ => seq(
      field('func', choice(
        'sigmoid', 'exp', 'log', 'abs', 'softplus',
        // Bayesian-modelling deterministic morphisms.
        'cumsum', 'softmax', 'cholesky_quad_form',
      )),
      '(',
      field('args', commaSep1($._let_arith)),
      ')',
    ),

    let_unary: $ => prec(PREC.let_unary, seq(
      '-',
      field('operand', $._let_atom),
    )),

    let_binop: $ => choice(
      prec.left(PREC.let_add, seq(
        field('left',  $._let_arith),
        field('op',    choice('+', '-')),
        field('right', $._let_arith),
      )),
      prec.left(PREC.let_mul, seq(
        field('left',  $._let_arith),
        field('op',    choice('*', '/')),
        field('right', $._let_arith),
      )),
    ),

    // ---------------------------------------------------------------
    // tokens
    // ---------------------------------------------------------------

    // `## …` doc comments are extracted into the AST and forwarded
    // into the program-theory schema metadata. Standalone `#` line
    // comments are dropped at parse time. The `##` form must be
    // matched before the bare `#` line_comment so the lexer doesn't
    // greedy-eat the second `#` as part of a regular comment.
    doc_comment:  _ => token(prec(1, seq('##', /[^\n]*/))),
    line_comment: _ => token(seq('#', /[^\n]*/)),

    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,

    integer: _ => /[0-9]+/,

    float:   _ => /[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?/,

    signed_number: $ => seq(optional('-'), choice($.integer, $.float)),
  },
});

/**
 * Comma-separated list with at least one element.
 * @param {RuleOrLiteral} rule
 * @returns {SeqRule}
 */
function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
