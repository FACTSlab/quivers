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

  extras: $ => [/\s/, $.line_comment],

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
      $.continuous_decl,
      $.stochastic_decl,
      $.discretize_decl,
      $.embed_decl,
      $.program_decl,
      $.let_decl,
      $.output_decl,
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

    output_decl: $ => seq('output', field('value', $._expr)),

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
    type_alias_decl: $ => seq(
      'type',
      field('name', $.identifier),
      '=',
      field('value', $._space_expr),
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

    program_decl: $ => seq(
      'program',
      field('name', $.identifier),
      optional(seq('(', field('params', commaSep1($.identifier)), ')')),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      field('steps', repeat1($._program_step)),
      'return',
      field('return', $._return_pattern),
    ),

    _program_step: $ => choice(
      $.draw_step,
      $.observe_step,
      $.arrow_draw_step,
      $.let_step,
    ),

    draw_step: $ => seq(
      'draw',
      field('vars', $._var_pattern),
      '~',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    ),

    observe_step: $ => seq(
      'observe',
      field('vars', $._var_pattern),
      '~',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    ),

    // alternative do-notation: `x <- f(...)`
    arrow_draw_step: $ => seq(
      field('var', $.identifier),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    ),

    let_step: $ => seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._let_arith),
    ),

    _draw_arg: $ => choice(
      $.identifier,
      $.signed_number,
    ),

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
      $.return_labeled_tuple,
    ),

    return_tuple: $ => seq(
      '(',
      commaSep1($.identifier),
      optional(','),
      ')',
    ),

    return_labeled_tuple: $ => seq(
      '(',
      commaSep1($.return_label_entry),
      optional(','),
      ')',
    ),

    return_label_entry: $ => seq(
      field('label', $.identifier),
      ':',
      field('var', $.identifier),
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
      $.let_var,
      $.let_literal,
    ),

    let_paren: $ => seq('(', $._let_arith, ')'),

    let_var: $ => $.identifier,

    let_literal: $ => $._numeric_literal,

    let_call: $ => seq(
      field('func', choice('sigmoid', 'exp', 'log', 'abs', 'softplus')),
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
