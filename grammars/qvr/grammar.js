/**
 * @file Quivers DSL grammar (0.11.0 homogenized surface)
 * @author Aaron Steven White <aaronstevenwhite@gmail.com>
 * @license MIT
 *
 * The 0.11.0 surface applies twelve homogenization moves:
 *
 *   1. Python-style indented blocks everywhere (INDENT / DEDENT via
 *      external scanner; tree-sitter-python style).
 *   2. ``KEYWORD NAME[(params)] : SIG [options] [body]`` is the
 *      single declaration header shape.
 *   3. ``composition NAME at LEVEL`` replaces algebra /
 *      semigroupoid / bilinear_form / composition_rule.
 *   4. ``morphism NAME : DOM -> COD [role=...]`` replaces latent /
 *      observed / kernel / embed / discretize.
 *   5. ``[k=v, ...]`` option block subsumes ``! effects``,
 *      ``depth N``, ``start S``, ``semiring R``, etc.
 *   6. ``~`` is the only initializer marker.
 *   7. ``## doc`` attaches to every declaration kind.
 *   8. ``type NAME : EXPR`` replaces object / space / alias /
 *      type-alias.
 *   9. ``[over=[...] [iid=...] [via=...]]`` unified.
 *  10. Every program step carries a leading keyword
 *      (sample / observe / marginalize / let / return).
 *  11. Effects move into the option block.
 *  12. Constructor-style sized types: ``FinSet(3)``,
 *      ``Euclidean(64)``; kernel rank moves into options.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  trans_compose: 1,
  compose: 1,
  tensor: 2,
  postfix: 3,
  type_coproduct: 1,
  type_slash: 2,
  type_product: 3,
  type_apply: 4,
  let_add: 1,
  let_mul: 2,
  let_unary: 3,
};

module.exports = grammar({
  name: 'qvr',

  /* Spaces and tabs are extras: ``extras`` are consumed silently
   * everywhere. NEWLINE is NOT an extra; it's a real token managed
   * by the external scanner. Doc comments and line comments are
   * extras so the parser can ignore them between tokens, but the
   * scanner suppresses comment lines from INDENT/DEDENT tracking. */
  extras: $ => [/[ \t]+/, $.doc_comment, $.line_comment],

  externals: $ => [
    $._newline,
    $._indent,
    $._dedent,
    $._eof,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    [$._let_atom, $.let_index],
  ],

  rules: {
    // -----------------------------------------------------------------
    // top level
    // -----------------------------------------------------------------

    source_file: $ => seq(repeat($._top), $._eof),

    /* Allow blank lines between top-level statements: the scanner
     * emits a NEWLINE for each, and we tolerate them here. */
    _top: $ => choice($._statement, $._newline),

    _statement: $ => choice(
      $.composition_decl,
      $.category_decl,
      $.rule_decl,
      $.schema_decl,
      $.type_decl,
      $.morphism_decl,
      $.bundle_decl,
      $.program_decl,
      $.contraction_decl,
      $.let_decl,
      $.export_decl,
      $.deduction_decl,
      $.signature_decl,
      $.encoder_decl,
      $.decoder_decl,
      $.loss_decl,
    ),

    // -----------------------------------------------------------------
    // composition (move #3)
    // -----------------------------------------------------------------

    composition_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'composition',
      field('name', $.identifier),
      optional(seq('at', field('level', $.composition_level))),
      optional(seq(
        ':',
        $._newline,
        $._indent,
        repeat($.composition_rule_entry),
        $._dedent,
      )),
      $._newline,
    ),

    composition_level: $ => choice(
      'algebra',
      'semigroupoid',
      'bilinear_form',
      'rule',
    ),

    composition_rule_entry: $ => seq(
      field('key', $.identifier),
      optional(seq(
        '(',
        field('params', commaSep1($.identifier)),
        ')',
      )),
      '=',
      field('body', $._let_arith),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // category
    // -----------------------------------------------------------------

    category_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'category',
      field('names', commaSep1($.identifier)),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // type (move #8)
    // -----------------------------------------------------------------

    type_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'type',
      field('name', $.identifier),
      ':',
      field('value', $._type_value),
      $._newline,
    ),

    _type_value: $ => choice(
      $.enum_set_literal,
      $.free_residuated_expr,
      $.free_monoid_expr,
      $._type_expr,
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
      optional(seq(',', commaSep1($.free_residuated_arg))),
      ')',
    ),

    free_residuated_arg: $ => choice(
      seq('depth', '=', field('depth', $.integer)),
      seq('ops', '=', '[', commaSep1(field('op', $.identifier)), ']'),
    ),

    free_monoid_expr: $ => seq(
      'FreeMonoid',
      '(',
      field('generators', $.identifier),
      ',',
      'max_length', '=', field('max_length', $.integer),
      ')',
    ),

    // -----------------------------------------------------------------
    // morphism (move #4 + #6)
    // -----------------------------------------------------------------

    morphism_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'morphism',
      field('name', $.identifier),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      field('options', $.option_block),
      optional(seq('~', field('init', $._morphism_init))),
      $._newline,
    ),

    _morphism_init: $ => choice(
      $.morphism_init_family,
      $._expr,
    ),

    morphism_init_family: $ => prec(2, seq(
      field('family', $.identifier),
      '(',
      optional(field('args', commaSep1($._draw_arg))),
      ')',
    )),

    // -----------------------------------------------------------------
    // bundle
    // -----------------------------------------------------------------

    bundle_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'bundle',
      field('name', $.identifier),
      '=',
      '[',
      optional(field('rules', commaSep1($.identifier))),
      ']',
      $._newline,
    ),

    // -----------------------------------------------------------------
    // contraction
    // -----------------------------------------------------------------

    contraction_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'contraction',
      field('name', $.identifier),
      '(',
      field('inputs', commaSep1($.contraction_input)),
      ')',
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      field('options', $.option_block),
      $._newline,
    ),

    contraction_input: $ => seq(
      field('name', $.identifier),
      ':',
      field('input_domain', $._type_expr),
      '->',
      field('input_codomain', $._type_expr),
    ),

    // -----------------------------------------------------------------
    // rule (top-level CCG/Lambek)
    // -----------------------------------------------------------------

    rule_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'rule',
      field('name', $.identifier),
      '(',
      field('variables', commaSep1($.identifier)),
      ')',
      ':',
      field('premises', commaSep1($._type_expr)),
      '=>',
      field('conclusion', $._type_expr),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // schema
    // -----------------------------------------------------------------

    schema_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'schema',
      field('name', $.identifier),
      '(',
      field('parameters', commaSep1($.schema_parameter)),
      ')',
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      $._newline,
    ),

    schema_parameter: $ => seq(
      field('names', commaSep1($.identifier)),
      ':',
      field('type', $._type_expr),
    ),

    // -----------------------------------------------------------------
    // let / export
    // -----------------------------------------------------------------

    let_decl: $ => prec.right(seq(
      optional(field('docs', $.doc_comment_group)),
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._expr),
      optional(seq(
        'where',
        ':',
        $._newline,
        $._indent,
        repeat1($.let_decl),
        $._dedent,
      )),
      $._newline,
    )),

    export_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'export',
      field('value', $._expr),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // deduction
    // -----------------------------------------------------------------

    deduction_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'deduction',
      field('name', $.identifier),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      optional(field('options', $.option_block)),
      ':',
      $._newline,
      $._indent,
      repeat($._deduction_body_entry),
      $._dedent,
      $._newline,
    ),

    _deduction_body_entry: $ => choice(
      $.deduction_atoms,
      $.deduction_rule,
      $.deduction_lexicon,
      $.deduction_lexicon_from_file,
    ),

    deduction_atoms: $ => seq(
      'atoms',
      field('atoms', commaSep1($.identifier)),
      $._newline,
    ),

    deduction_rule: $ => seq(
      'rule',
      field('name', $.identifier),
      ':',
      field('premises', commaSep1($._type_expr)),
      choice('|-', '⊢'),
      field('conclusion', $._type_expr),
      $._newline,
    ),

    deduction_lexicon: $ => seq(
      'lexicon',
      ':',
      $._newline,
      $._indent,
      repeat($.lexicon_entry),
      $._dedent,
      $._newline,
    ),

    lexicon_entry: $ => seq(
      field('word', $.string),
      ':',
      field('category', $._type_expr),
      '=',
      field('lf', $._let_arith),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    deduction_lexicon_from_file: $ => seq(
      'lexicon',
      'from',
      field('path', $.string),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // signature
    // -----------------------------------------------------------------

    signature_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'signature',
      field('name', $.identifier),
      optional(seq('(', field('params', commaSep1($.identifier)), ')')),
      ':',
      $._newline,
      $._indent,
      repeat($._signature_body_entry),
      $._dedent,
      $._newline,
    ),

    _signature_body_entry: $ => choice(
      $.signature_sorts,
      $.signature_constructors,
      $.signature_binders,
      $.signature_vertex_kinds,
      $.signature_edge_kinds,
    ),

    signature_sorts: $ => seq(
      'sorts',
      ':',
      $._newline,
      $._indent,
      repeat($.sort_decl),
      $._dedent,
      $._newline,
    ),

    sort_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    sort_kind: $ => choice('object', 'index', 'data'),

    vocab_literal: $ => choice($.string, $.integer, $.float),

    signature_constructors: $ => seq(
      'constructors',
      ':',
      $._newline,
      $._indent,
      repeat($.constructor_decl),
      $._dedent,
      $._newline,
    ),

    constructor_decl: $ => seq(
      field('name', $.identifier),
      ':',
      optional(field('domain', commaSep1($._sig_sort))),
      '->',
      field('codomain', $._sig_sort),
      $._newline,
    ),

    _sig_sort: $ => prec(1, $.identifier),

    signature_binders: $ => seq(
      'binders',
      ':',
      $._newline,
      $._indent,
      repeat($.binder_decl),
      $._dedent,
      $._newline,
    ),

    binder_decl: $ => seq(
      field('name', $.identifier),
      ':',
      'binds',
      '(',
      field('binds', commaSep1($.binder_var_decl)),
      ')',
      'in',
      '(',
      field('scoped', commaSep1($.binder_arg_decl)),
      ')',
      '->',
      field('codomain', $._sig_sort),
      $._newline,
    ),

    binder_var_decl: $ => seq(
      field('var', $.identifier),
      ':',
      field('sort', $.identifier),
      optional(seq(
        ':',
        field('annot', $.identifier),
        ':',
        field('annot_sort', $.identifier),
      )),
    ),

    binder_arg_decl: $ => seq(
      field('arg', $.identifier),
      ':',
      field('sort', $.identifier),
    ),

    signature_vertex_kinds: $ => seq(
      'vertex_kinds',
      ':',
      $._newline,
      $._indent,
      repeat($.vertex_kind_decl),
      $._dedent,
      $._newline,
    ),

    vertex_kind_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    signature_edge_kinds: $ => seq(
      'edge_kinds',
      ':',
      $._newline,
      $._indent,
      repeat($.edge_kind_decl),
      $._dedent,
      $._newline,
    ),

    edge_kind_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('src', $.identifier),
      field('arrow', $.edge_arrow),
      field('tgt', $.identifier),
      $._newline,
    ),

    edge_arrow: $ => choice('->', '--'),

    // -----------------------------------------------------------------
    // encoder / decoder / loss
    // -----------------------------------------------------------------

    encoder_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'encoder',
      field('name', $.identifier),
      ':',
      field('signature', $.identifier),
      optional(seq(
        '(',
        field('sig_args', commaSep1($.identifier)),
        ')',
      )),
      optional(field('options', $.option_block)),
      optional(seq(
        ':',
        $._newline,
        $._indent,
        repeat($._encoder_body_entry),
        $._dedent,
      )),
      $._newline,
    ),

    _encoder_body_entry: $ => choice(
      $.encoder_dim,
      $.encoder_iterations,
      $.encoder_readout,
      $.encoder_op_rule,
      $.encoder_message_rule,
      $.encoder_update_rule,
      $.encoder_init_rule,
      $.encoder_var_init,
    ),

    encoder_dim: $ => seq(
      'dim',
      field('sort', $.identifier),
      '=',
      field('dim', $.integer),
      $._newline,
    ),

    encoder_iterations: $ => seq(
      'iterations',
      field('iterations', $.integer),
      $._newline,
    ),

    encoder_readout: $ => seq(
      'readout',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    encoder_op_rule: $ => seq(
      field('op', $.identifier),
      optional(seq('(', commaSep1(field('args', $.identifier)), ')')),
      optional(choice(
        seq('recurrent', field('state', $.identifier)),
        seq('attention', field('prefix', $.identifier)),
      )),
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    encoder_init_rule: $ => seq(
      'init',
      field('kind', $.identifier),
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    encoder_message_rule: $ => seq(
      'message',
      '[',
      field('edge_kind', $.identifier),
      ']',
      '(',
      field('src', $.identifier),
      ',',
      field('tgt', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    encoder_update_rule: $ => seq(
      'update',
      '[',
      field('vertex_kind', $.identifier),
      ']',
      '(',
      field('self', $.identifier),
      ',',
      field('msgs', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    encoder_var_init: $ => seq(
      'var_init',
      field('var_sort', $.identifier),
      optional(seq(
        'from',
        field('annot_sort', $.identifier),
        optional(seq('as', field('ty', $.identifier))),
      )),
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    decoder_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'decoder',
      field('name', $.identifier),
      'over',
      field('signature', $.identifier),
      optional(seq('(', field('sig_args', commaSep1($.identifier)), ')')),
      optional(field('options', $.option_block)),
      ':',
      $._newline,
      $._indent,
      repeat($._decoder_body_entry),
      $._dedent,
      $._newline,
    ),

    _decoder_body_entry: $ => choice(
      $.decoder_dim,
      $.decoder_structure,
      $.decoder_primitive,
      $.decoder_factor,
      $.decoder_binder_select,
      $.decoder_body_default,
    ),

    decoder_dim: $ => seq(
      'dim',
      field('sort', $.identifier),
      '=',
      field('dim', $.integer),
      $._newline,
    ),

    decoder_structure: $ => seq(
      'structure',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    decoder_primitive: $ => seq(
      'primitive',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    decoder_factor: $ => seq(
      'factor',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    decoder_binder_select: $ => seq(
      'binder_select',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
      $._newline,
    ),

    decoder_body_default: $ => seq(
      'body',
      '|->',
      field('default', 'recursive'),
      $._newline,
    ),

    loss_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'loss',
      field('name', $.identifier),
      optional(field('options', $.option_block)),
      ':',
      $._newline,
      $._indent,
      field('body', $._let_arith),
      $._newline,
      $._dedent,
      $._newline,
    ),

    // -----------------------------------------------------------------
    // program (indented body, leading-keyword steps)
    // -----------------------------------------------------------------

    program_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'program',
      field('name', $.identifier),
      optional(seq('(', field('params', commaSep1($._program_param)), ')')),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      optional(field('options', $.option_block)),
      ':',
      $._newline,
      $._indent,
      field('steps', repeat($._program_step)),
      $.return_step,
      $._dedent,
      $._newline,
    ),

    _program_param: $ => choice(
      $.identifier,
      $.typed_program_param,
    ),

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
      $.sample_step,
      $.observe_step,
      $.marginalize_step,
      $.let_step,
    ),

    sample_step: $ => seq(
      'sample',
      field('vars', $._var_pattern),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    observe_step: $ => seq(
      'observe',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    marginalize_step: $ => seq(
      'marginalize',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
      optional(field('options', $.option_block)),
      ':',
      $._newline,
      $._indent,
      field('scope', repeat($._program_step)),
      $._dedent,
      $._newline,
    ),

    let_step: $ => seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._let_arith),
      $._newline,
    ),

    return_step: $ => seq(
      'return',
      field('return', $._return_pattern),
      $._newline,
    ),

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

    _var_pattern: $ => choice($.identifier, $.var_tuple),

    var_tuple: $ => seq(
      '[',
      commaSep1($.identifier),
      optional(','),
      ']',
    ),

    _return_pattern: $ => choice(
      $.identifier,
      $.return_labeled_tuple,
      $.return_tuple,
    ),

    return_tuple: $ => seq(
      '(',
      commaSep1($.identifier),
      optional(','),
      ')',
    ),

    return_labeled_tuple: $ => prec(1, seq(
      '(',
      commaSep1($.return_label_entry),
      optional(','),
      ')',
    )),

    return_label_entry: $ => seq(
      field('label', $.identifier),
      ':',
      field('var', $.identifier),
    ),

    // -----------------------------------------------------------------
    // type expressions
    // -----------------------------------------------------------------

    _type_expr: $ => choice(
      $.type_coproduct,
      $.type_slash,
      $.type_product,
      $.discrete_constructor,
      $.continuous_constructor,
      $.type_effect_apply,
      $.type_atom,
      $.type_paren,
    ),

    type_atom: $ => choice($.identifier, $.integer),
    type_paren: $ => seq('(', $._type_expr, ')'),

    type_product: $ => prec.left(PREC.type_product, seq(
      field('left', $._type_expr),
      '*',
      field('right', $._type_expr),
    )),

    type_coproduct: $ => prec.left(PREC.type_coproduct, seq(
      field('left', $._type_expr),
      '+',
      field('right', $._type_expr),
    )),

    type_slash: $ => prec.left(PREC.type_slash, seq(
      field('result', $._type_expr),
      field('direction', choice('/', '\\')),
      field('argument', $._type_expr),
    )),

    type_effect_apply: $ => prec(PREC.type_apply, seq(
      field('effect', $.identifier),
      '(',
      field('args', commaSep1($._type_expr)),
      ')',
    )),

    /* Constructor calls for sized types. The grammar keeps a single
     * call shape ``Name(args)`` but tags each kind so downstream
     * code dispatches on the parse-tree node, not on the
     * constructor name string. Operators that combine discrete and
     * continuous (``FinSet(N) * Euclidean(D)`` is a legitimate
     * mixed product) remain in the unified ``_type_expr`` family;
     * categorical validity is a type-checking concern handled by
     * the compiler, not the grammar. */
    discrete_constructor: $ => prec(PREC.type_apply, seq(
      field('constructor', 'FinSet'),
      '(',
      optional(field('args', commaSep1($._type_constructor_arg))),
      ')',
    )),

    /* Continuous-space constructors:
     *
     *   Real(N)                       ℝ^N (unbounded)
     *   Real(N, low=L)                ℝ^N restricted to x >= L (per dim)
     *   Real(N, low=L, high=H)        the box [L, H]^N
     *   Real(N, high=H)               x <= H (per dim)
     *
     *   Simplex(K)                    the (K-1)-simplex (components sum to 1)
     *   CholeskyFactor(D)             lower-triangular with positive diagonal
     *
     * Product spaces use the ``*`` operator on type expressions:
     * ``Real(64) * Real(32)`` instead of a dedicated ``ProductSpace``
     * constructor. The historical PositiveReals and UnitInterval
     * special-cases are subsumed by ``Real(N, low=...)`` and
     * ``Real(N, low=..., high=...)`` respectively. */
    continuous_constructor: $ => prec(PREC.type_apply, seq(
      field('constructor', choice(
        'Real',
        'Simplex',
        'Sphere',
        'Ball',
        'CholeskyFactor',
        'Covariance',
        'Correlation',
        'Orthogonal',
        'Stiefel',
        'LowerTriangular',
        'Diagonal',
      )),
      '(',
      optional(field('args', commaSep1($._type_constructor_arg))),
      ')',
    )),

    _type_constructor_arg: $ => choice(
      $.type_constructor_kwarg,
      $.integer,
      $.float,
      $.identifier,
    ),

    type_constructor_kwarg: $ => seq(
      field('key', $.identifier),
      '=',
      field('value', $._numeric_literal),
    ),

    _numeric_literal: $ => choice($.integer, $.float),

    // -----------------------------------------------------------------
    // option block (move #5)
    // -----------------------------------------------------------------

    option_block: $ => seq(
      '[',
      commaSep1($.option_entry),
      ']',
    ),

    option_entry: $ => seq(
      field('key', $.identifier),
      optional(seq('=', field('value', $._option_value))),
    ),

    _option_value: $ => choice(
      $.option_list,
      $.option_call,
      $.identifier,
      $.integer,
      $.float,
      $.string,
    ),

    option_call: $ => prec(2, seq(
      field('func', $.identifier),
      '(',
      optional(field('args', commaSep1(choice(
        $.string, $.integer, $.float, $.identifier,
      )))),
      ')',
    )),

    option_list: $ => seq(
      '[',
      optional(commaSep1(field('item', choice(
        $.identifier, $.string, $.integer, $.float,
      )))),
      ']',
    ),

    // -----------------------------------------------------------------
    // value (morphism) expressions
    // -----------------------------------------------------------------

    _expr: $ => choice(
      $.trans_compose,
      $.compose_expr,
      $.tensor_expr,
      $.postfix_expr,
      $._atom_expr,
    ),

    trans_compose: $ => prec.left(PREC.trans_compose, seq(
      field('left', $._expr),
      '>>>',
      field('right', $._expr),
    )),

    compose_expr: $ => prec.left(PREC.compose, seq(
      field('left', $._expr),
      field('op', choice(
        '>>', '<<', '>=>',
        '*>', '~>', '||>', '?>', '&&>', '+>',
        '$>', '%>',
      )),
      field('right', $._expr),
    )),

    tensor_expr: $ => prec.left(PREC.tensor, seq(
      field('left', $._expr),
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
      seq(field('name', choice('curry_right', 'curry_left'))),
      seq(
        field('name', 'change_base'),
        '(',
        field('arg', $._expr),
        ')',
      ),
      seq(field('name', 'dagger')),
      seq(
        field('name', 'trace'),
        '(',
        field('args', $.identifier),
        ')',
      ),
      seq(field('name', 'freeze')),
    ),

    _atom_expr: $ => choice(
      $.expr_paren,
      $.identity_expr,
      $.cup_expr,
      $.cap_expr,
      $.from_data_expr,
      $.fan_expr,
      $.repeat_expr,
      $.stack_expr,
      $.scan_expr,
      $.parser_expr,
      $.chart_fold_expr,
      $.morphism_call,
      $.expr_ident,
    ),

    morphism_call: $ => prec(20, seq(
      field('callee', $.identifier),
      '(',
      field('args', commaSep1($.identifier)),
      ')',
    )),

    expr_paren: $ => seq('(', $._expr, ')'),
    expr_ident: $ => $.identifier,

    identity_expr: $ => seq('identity', '(', field('object', $.identifier), ')'),
    cup_expr: $ => seq('cup', '(', field('object', $.identifier), ')'),
    cap_expr: $ => seq('cap', '(', field('object', $.identifier), ')'),

    from_data_expr: $ => seq(
      'from_data',
      '(',
      field('key', $._string_literal),
      ')',
    ),

    fan_expr: $ => seq('fan', '(', field('args', commaSep1($._expr)), ')'),

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

    chart_fold_expr: $ => seq(
      'chart_fold',
      '(',
      optional(field('args', commaSep1($.chart_fold_arg))),
      ')',
    ),

    chart_fold_arg: $ => prec(10, seq(
      field('key', choice('lex', 'binary', 'unary', 'start', 'depth', 'effect_depth')),
      '=',
      field('value', choice($._expr, $.integer)),
    )),

    parser_arg: $ => seq(
      field('key', choice('rules', 'categories', 'terminal', 'start', 'depth', 'constructors')),
      '=',
      field('value', choice($.ident_list, $.identifier, $.integer)),
    ),

    ident_list: $ => seq('[', optional(commaSep1($.identifier)), ']'),

    // -----------------------------------------------------------------
    // let-arithmetic
    // -----------------------------------------------------------------

    _let_arith: $ => choice($.let_binop, $.let_unary, $._let_atom),

    _let_atom: $ => choice(
      $.let_paren,
      $.let_method_call,
      $.let_call,
      $.let_index,
      $.let_list,
      $.let_factor,
      $.let_lambda,
      $.let_string,
      $.let_var,
      $.let_literal,
    ),

    let_factor: $ => prec.right(seq(
      'factor',
      commaSep1(field('binders', $.let_factor_binder)),
      'in',
      choice(
        seq(
          '{',
          commaSep1(field('cases', $.let_factor_case)),
          optional(','),
          '}',
        ),
        field('body', $._let_arith),
      ),
    )),

    let_factor_binder: $ => seq(
      field('var', $.identifier),
      ':',
      field('index', $._type_expr),
    ),

    let_factor_case: $ => seq(
      field('label', $.integer),
      '->',
      field('value', $._let_arith),
    ),

    let_list: $ => seq(
      '[',
      optional(seq(commaSep1($._let_arith), optional(','))),
      ']',
    ),

    let_string: $ => $._string_literal,

    let_lambda: $ => prec.right(seq(
      field('param', $.identifier),
      '->',
      field('body', $._let_arith),
    )),

    let_method_call: $ => prec.left(2, seq(
      field('receiver', $.let_var),
      '.',
      field('method', $.identifier),
      '(',
      optional(field('args', commaSep1($._let_arith))),
      ')',
    )),

    let_index: $ => prec.left(seq(
      field('array', $.let_var),
      '[',
      field('indices', commaSep1($._let_arith)),
      ']',
    )),

    let_paren: $ => seq('(', $._let_arith, ')'),
    let_var: $ => $.identifier,
    let_literal: $ => $._numeric_literal,

    let_call: $ => prec(1, seq(
      field('func', $.identifier),
      '(',
      optional(field('args', commaSep1($._let_arith))),
      ')',
    )),

    let_unary: $ => prec(PREC.let_unary, seq(
      '-',
      field('operand', $._let_atom),
    )),

    let_binop: $ => choice(
      prec.left(PREC.let_add, seq(
        field('left', $._let_arith),
        field('op', choice('+', '-')),
        field('right', $._let_arith),
      )),
      prec.left(PREC.let_mul, seq(
        field('left', $._let_arith),
        field('op', choice('*', '/')),
        field('right', $._let_arith),
      )),
    ),

    // -----------------------------------------------------------------
    // doc-comment groups
    // -----------------------------------------------------------------

    doc_comment_group: $ => repeat1($.doc_comment),

    // -----------------------------------------------------------------
    // tokens
    // -----------------------------------------------------------------

    doc_comment: _ => token(prec(1, seq('##', /[^\n]*/))),
    line_comment: _ => token(seq('#', /[^\n]*/)),

    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,
    integer: _ => /[0-9]+/,
    float: _ => /[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?/,
    signed_number: $ => seq(optional('-'), choice($.integer, $.float)),

    _string_literal: $ => $.string,
    string: _ => token(seq(
      '"',
      repeat(choice(
        /[^"\\\n]/,
        seq('\\', /./),
      )),
      '"',
    )),
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
