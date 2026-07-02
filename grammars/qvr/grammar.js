/**
 * @file Quivers DSL grammar
 * @author Aaron Steven White <aaronstevenwhite@gmail.com>
 * @license MIT
 *
 * Surface invariants:
 *
 *   1. Python-style indented blocks everywhere (INDENT / DEDENT via
 *      external scanner; tree-sitter-python style).
 *   2. ``KEYWORD NAMES[(params)] : SIG [options] [body]`` is the
 *      single declaration header shape. NAMES is a comma-separated
 *      identifier list wherever a declaration admits families
 *      (category / object / morphism); the option block is optional
 *      on every declaration that takes one.
 *   3. ``[k=v, ...]`` option blocks carry every declaration-level
 *      knob (role, scale, level, semiring, effects, over, iid,
 *      via, ...). Option values admit signed numbers.
 *   4. Sized spaces use space application (``FinSet 3``,
 *      ``Real 28 28``); constructor keyword options use braces
 *      (``Real 1 {low=-1.0, high=1.0}``), so a trailing ``[...]``
 *      always belongs to the enclosing declaration.
 *   5. ``~`` is the only initializer marker; ``#!`` doc comments
 *      attach to every declaration kind.
 *   6. Every program step carries a leading keyword
 *      (sample / observe / marginalize / let / score / return);
 *      variable patterns are parenthesized tuples or a bare name.
 *   7. ``|-`` (or ``⊢``) is the only premises-to-conclusion
 *      turnstile, for top-level rules and deduction rules alike.
 *   8. Top-level ``define`` binds morphism expressions; program-step
 *      ``let`` binds tensor arithmetic. The two sublanguages never
 *      share a keyword.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  trans_compose: 1,
  compose: 1,
  tensor: 2,
  postfix: 3,
  object_coproduct: 1,
  object_slash: 2,
  object_product: 3,
  object_apply: 4,
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
  extras: $ => [
    /[ \t]+/,
    $.line_comment,
    $.block_comment,
  ],

  externals: $ => [
    $._newline,
    $._indent,
    $._dedent,
    $._eof,
  ],

  word: $ => $.identifier,

  conflicts: _ => [],

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
      $.object_decl,
      $.morphism_decl,
      $.bundle_decl,
      $.program_decl,
      $.contraction_decl,
      $.define_decl,
      $.export_decl,
      $.deduction_decl,
      $.signature_decl,
      $.encoder_decl,
      $.decoder_decl,
      $.loss_decl,
      $.pragma_outer,
      $.pragma_inner,
    ),

    // -----------------------------------------------------------------
    // pragmas: ``#[ k = v, ... ]`` outer, ``#![ k = v, ... ]`` inner.
    // Top-level statements; the compiler decides attachment (next
    // decl vs. module-level).
    // -----------------------------------------------------------------

    pragma_outer: $ => seq(
      '#[',
      field('entries', commaSep1($.pragma_entry)),
      ']',
      $._newline,
    ),

    pragma_inner: $ => seq(
      '#![',
      field('entries', commaSep1($.pragma_entry)),
      ']',
      $._newline,
    ),

    pragma_entry: $ => seq(
      field('key', $.identifier),
      optional(seq('=', field('value', $._option_value))),
    ),

    // -----------------------------------------------------------------
    // composition
    // -----------------------------------------------------------------

    composition_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'composition',
      field('name', $.identifier),
      optional(field('options', $.option_block)),
      choice(
        seq(
          $._newline,
          $._indent,
          repeat1(choice($.composition_rule_entry, $._newline)),
          $._dedent,
        ),
        $._newline,
      ),
    ),

    composition_rule_entry: $ => seq(
      field('key', $.identifier),
      optional(seq(
        bracketedList($, '(', field('params', $.identifier), ')'),
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
    // object
    // -----------------------------------------------------------------

    /* ``object A, B : V`` declares one object per name, each with the
     * same value expression. */
    object_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'object',
      field('names', commaSep1($.identifier)),
      ':',
      field('value', $._object_value),
      $._newline,
    ),

    _object_value: $ => choice(
      $.enum_set_literal,
      $.free_residuated_expr,
      $.free_monoid_expr,
      $._object_expr,
    ),

    enum_set_literal: $ => seq(
      bracketedList($, '{', field('elements', $.identifier), '}'),
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
    // morphism
    // -----------------------------------------------------------------

    /* ``morphism f, g : A -> B`` declares one morphism per name, each
     * with the same signature and options but independent parameters.
     * The option block is optional; ``role`` defaults by inference
     * (sampled -> latent, observed -> observed, otherwise kernel). */
    morphism_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'morphism',
      field('names', commaSep1($.identifier)),
      ':',
      field('domain', $._object_expr),
      '->',
      field('codomain', $._object_expr),
      optional(field('options', $.option_block)),
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
      ':',
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
      bracketedList($, '(', field('inputs', $.contraction_input), ')'),
      ':',
      field('domain', $._object_expr),
      '->',
      field('codomain', $._object_expr),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    contraction_input: $ => seq(
      field('name', $.identifier),
      ':',
      field('input_domain', $._object_expr),
      '->',
      field('input_codomain', $._object_expr),
    ),

    // -----------------------------------------------------------------
    // rule (top-level CCG/Lambek)
    // -----------------------------------------------------------------

    rule_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'rule',
      field('name', $.identifier),
      bracketedList($, '(', field('variables', $.identifier), ')'),
      ':',
      field('premises', commaSep1($._object_expr)),
      choice('|-', '⊢'),
      field('conclusion', $._object_expr),
      $._newline,
    ),

    // -----------------------------------------------------------------
    // schema
    // -----------------------------------------------------------------

    schema_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'schema',
      field('name', $.identifier),
      bracketedList($, '(', field('parameters', $.schema_parameter), ')'),
      ':',
      field('domain', $._object_expr),
      '->',
      field('codomain', $._object_expr),
      $._newline,
    ),

    schema_parameter: $ => seq(
      field('names', commaSep1($.identifier)),
      ':',
      field('type', $._object_expr),
    ),

    // -----------------------------------------------------------------
    // define / export
    // -----------------------------------------------------------------

    /* ``define`` binds a morphism expression at the top level. The
     * program-step ``let`` binds tensor arithmetic inside program
     * bodies; the two binding forms never share a keyword. */
    define_decl: $ => prec.right(seq(
      optional(field('docs', $.doc_comment_group)),
      'define',
      field('name', $.identifier),
      '=',
      field('value', $._expr),
      choice(
        seq(
          'where',        $._newline,
          $._indent,
          repeat1(choice($.define_decl, $._newline)),
          $._dedent,
        ),
        $._newline,
      ),
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
      field('domain', $._object_expr),
      '->',
      field('codomain', $._object_expr),
      optional(field('options', $.option_block)),      $._newline,
      $._indent,
      repeat1(choice($._deduction_body_entry, $._newline)),
      $._dedent,
    ),

    _deduction_body_entry: $ => choice(
      $.deduction_atoms,
      $.deduction_binders,
      $.deduction_rule,
      $.deduction_lexicon,
      $.deduction_lexicon_from_file,
    ),

    deduction_atoms: $ => seq(
      'atoms',
      field('atoms', commaSep1($.identifier)),
      $._newline,
    ),

    // ``binders`` declares constructors whose first argument is
    // a bound variable. The compiler treats binder applications
    // specially: the variable position is alpha-renamed to a
    // fresh canonical symbol per term construction so the chart
    // identifies alpha-equivalent terms, and pattern-matching on
    // a binder's bound variable correctly threads the renaming
    // through the rule's bindings.
    deduction_binders: $ => seq(
      'binders',
      field('binders', commaSep1($.identifier)),
      $._newline,
    ),

    deduction_rule: $ => seq(
      'rule',
      field('name', $.identifier),
      ':',
      field('premises', commaSep1($._object_expr)),
      choice('|-', '⊢'),
      field('conclusion', $._object_expr),
      optional(field('pragma', $.lexicon_pragma)),
      $._newline,
    ),

    deduction_lexicon: $ => seq(
      'lexicon',      $._newline,
      $._indent,
      repeat1(choice($.lexicon_entry, $._newline)),
      $._dedent,
    ),

    /* ``"a", "an" : Det = LF`` declares one entry per word, each with
     * the same category and logical form. */
    lexicon_entry: $ => seq(
      field('words', commaSep1($.string)),
      ':',
      field('category', $._lexicon_category),
      '=',
      field('lf', $._let_arith),
      optional(field('pragma', $.lexicon_pragma)),
      $._newline,
    ),

    // Inline pragma form used as a trailing attribute on lexicon
    // entries. Distinct from ``pragma_outer`` only in that it does
    // not terminate with a newline (the enclosing lexicon_entry
    // owns the trailing newline). ``#`` opens the comment / pragma
    // family, so no let-arith expression can extend past the lf
    // into this position; the surface is unambiguous.
    lexicon_pragma: $ => seq(
      '#[',
      field('entries', commaSep1($.pragma_entry)),
      ']',
    ),

    _lexicon_category: $ => choice(
      '*',
      $.enum_set_literal,
      $._object_expr,
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
      optional(bracketedList($, '(', field('params', $.identifier), ')')),      $._newline,
      $._indent,
      repeat1(choice($._signature_body_entry, $._newline)),
      $._dedent,
    ),

    _signature_body_entry: $ => choice(
      $.signature_sorts,
      $.signature_constructors,
      $.signature_binders,
      $.signature_vertex_kinds,
      $.signature_edge_kinds,
    ),

    signature_sorts: $ => seq(
      'sorts',      $._newline,
      $._indent,
      repeat1(choice($.sort_decl, $._newline)),
      $._dedent,
    ),

    sort_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    sort_kind: $ => choice('object', 'index', 'data'),

    signature_constructors: $ => seq(
      'constructors',      $._newline,
      $._indent,
      repeat1(choice($.constructor_decl, $._newline)),
      $._dedent,
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
      'binders',      $._newline,
      $._indent,
      repeat1(choice($.binder_decl, $._newline)),
      $._dedent,
    ),

    binder_decl: $ => seq(
      field('name', $.identifier),
      ':',
      'binds',
      bracketedList($, '(', field('binds', $.binder_var_decl), ')'),
      'in',
      bracketedList($, '(', field('scoped', $.binder_arg_decl), ')'),
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
      'vertex_kinds',      $._newline,
      $._indent,
      repeat1(choice($.vertex_kind_decl, $._newline)),
      $._dedent,
    ),

    vertex_kind_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    signature_edge_kinds: $ => seq(
      'edge_kinds',      $._newline,
      $._indent,
      repeat1(choice($.edge_kind_decl, $._newline)),
      $._dedent,
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
        bracketedList($, '(', field('sig_args', $.identifier), ')'),
      )),
      optional(field('options', $.option_block)),
      choice(
        seq(
          $._newline,
          $._indent,
          repeat1(choice($._encoder_body_entry, $._newline)),
          $._dedent,
        ),
        $._newline,
      ),
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

    /* Every encoder body entry is keyword-led; ``op`` introduces a
     * constructor rewrite rule, so an operator may carry any name
     * (including ``dim`` or ``init``) without shadowing the sibling
     * entry keywords. */
    encoder_op_rule: $ => seq(
      'op',
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
      ':',
      field('signature', $.identifier),
      optional(bracketedList($, '(', field('sig_args', $.identifier), ')')),
      optional(field('options', $.option_block)),      $._newline,
      $._indent,
      repeat1(choice($._decoder_body_entry, $._newline)),
      $._dedent,
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
      optional(field('options', $.option_block)),      $._newline,
      $._indent,
      field('body', $._let_arith),
      $._newline,
      $._dedent,
    ),

    // -----------------------------------------------------------------
    // program (indented body, leading-keyword steps)
    // -----------------------------------------------------------------

    program_decl: $ => seq(
      optional(field('docs', $.doc_comment_group)),
      'program',
      field('name', $.identifier),
      optional(bracketedList($, '(', field('params', $._program_param), ')')),
      ':',
      field('domain', $._object_expr),
      '->',
      field('codomain', $._object_expr),
      optional(field('options', $.option_block)),
      $._newline,
      $._indent,
      repeat(choice(field('steps', $._program_step), $._newline)),
      $.return_step,
      $._dedent,
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
      field('domain', $._object_expr),
      ',',
      field('codomain', $._object_expr),
      ']',
    ),

    _program_step: $ => choice(
      $.sample_step,
      $.observe_step,
      $.marginalize_step,
      $.let_step,
      $.score_step,
    ),

    sample_step: $ => seq(
      'sample',
      field('vars', $._var_pattern),
      optional(seq(':', field('index', $._object_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(bracketedList($, '(', field('args', $._draw_arg), ')')),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    observe_step: $ => seq(
      'observe',
      field('vars', $._var_pattern),
      optional(seq(':', field('index', $._object_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(bracketedList($, '(', field('args', $._draw_arg), ')')),
      optional(field('options', $.option_block)),
      $._newline,
    ),

    marginalize_step: $ => seq(
      'marginalize',
      field('var', $.identifier),
      optional(seq(':', field('index', $._object_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(bracketedList($, '(', field('args', $._draw_arg), ')')),
      optional(field('options', $.option_block)),
      $._newline,
      $._indent,
      repeat1(choice(field('scope', $._program_step), $._newline)),
      $._dedent,
    ),

    let_step: $ => seq(
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._let_arith),
      $._newline,
    ),

    // Score / factor step: ``score NAME = EXPR``. The value of
    // ``EXPR`` is added to the program's ``log_joint`` (the
    // deduction analog of an ``observe`` whose log-density comes
    // from a chart goal weight or any other tensor expression).
    score_step: $ => seq(
      'score',
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
      $.family_call_arg,
      $.list_arg,
      $.identifier,
      $.signed_number,
    ),

    // Distribution-call expression at a draw-arg position, e.g.
    // `Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])`. The
    // compiler recurses into `args` to build the inner distribution
    // before passing it to the outer family's builder. See the
    // measure-algebra design note for the operator vocabulary.
    family_call_arg: $ => prec(2, seq(
      field('family', $.identifier),
      '(',
      optional(field('args', commaSep1($._draw_arg))),
      ')',
    )),

    // List-of-draw-args, e.g. `[0.3, 0.7]` for mixture weights or
    // `[PointMass(0), Poisson(rate)]` for mixture components. Mirrors
    // the let-expression list literal but at the draw-arg position.
    list_arg: $ => seq(
      '[',
      optional(commaSep1($._draw_arg)),
      ']',
    ),

    bracket_index_arg: $ => prec(1, seq(
      field('name', $.identifier),
      '[',
      field('index', $._object_expr),
      ']',
    )),

    /* Variable patterns share the parenthesized-tuple shape with
     * return patterns: a bare name or ``(a, b)``. */
    _var_pattern: $ => choice($.identifier, $.var_tuple),

    var_tuple: $ => seq(
      '(',
      commaSep1($.identifier),
      optional(','),
      ')',
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

    _object_expr: $ => choice(
      $.object_coproduct,
      $.object_slash,
      $.object_product,
      $.discrete_constructor,
      $.continuous_constructor,
      $.object_effect_apply,
      $.object_atom,
      $.object_paren,
    ),

    object_atom: $ => $.identifier,
    object_paren: $ => seq('(', $._object_expr, ')'),

    object_product: $ => prec.left(PREC.object_product, seq(
      field('left', $._object_expr),
      '*',
      field('right', $._object_expr),
    )),

    object_coproduct: $ => prec.left(PREC.object_coproduct, seq(
      field('left', $._object_expr),
      '+',
      field('right', $._object_expr),
    )),

    object_slash: $ => prec.left(PREC.object_slash, seq(
      field('result', $._object_expr),
      field('direction', choice('/', '\\')),
      field('argument', $._object_expr),
    )),

    object_effect_apply: $ => prec(PREC.object_apply, seq(
      field('effect', $.identifier),
      bracketedList($, '(', field('args', $._object_expr), ')'),
    )),

    /* Sized-space constructors use space application, matching the
     * mathematical convention that ``FinSet`` names the category and
     * ``FinSet N`` its canonical n-element object. Operators that
     * combine discrete and continuous (``FinSet N * Real D`` is a
     * legitimate mixed product) remain in the unified
     * ``_object_expr`` family; categorical validity is a
     * type-checking concern handled by the compiler, not the
     * grammar. */
    discrete_constructor: $ => prec(PREC.object_apply, seq(
      field('constructor', 'FinSet'),
      field('cardinality', choice($.integer, $.identifier)),
    )),

    /* Continuous-space constructors take space-separated positional
     * args, mirroring ``FinSet N``, with keyword options in a
     * trailing brace block:
     *
     *   Real 64                       -- one-dim real vector space
     *   Real 28 28                    -- 2D tensor space
     *   Real 1 {low=0.0}              -- half-line
     *   Real 1 {low=-1.0, high=1.0}   -- the box [-1, 1]
     *   Simplex 10                    -- the (K-1)-simplex
     *   CholeskyFactor 4              -- lower-triangular w/ positive diagonal
     *
     * Braces keep constructor options disjoint from declaration
     * option blocks: a trailing ``[...]`` always belongs to the
     * enclosing declaration. Product spaces use the ``*`` operator
     * on type expressions (``Real 64 * Real 32``). */
    continuous_constructor: $ => prec(PREC.object_apply, seq(
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
      repeat1(field('args', $._object_constructor_arg)),
      optional(field('options', $.constructor_options)),
    )),

    _object_constructor_arg: $ => choice(
      $.integer,
      $.float,
      $.identifier,
    ),

    constructor_options: $ => bracketedList($, '{', $.constructor_kwarg, '}'),

    constructor_kwarg: $ => seq(
      field('key', $.identifier),
      '=',
      field('value', choice($.signed_number, $.identifier)),
    ),

    _numeric_literal: $ => choice($.integer, $.float),

    // -----------------------------------------------------------------
    // option block
    // -----------------------------------------------------------------

    option_block: $ => bracketedList($, '[', $.option_entry, ']'),

    option_entry: $ => seq(
      field('key', $.identifier),
      optional(seq('=', field('value', $._option_value))),
    ),

    /* Numeric option values are signed, mirroring draw arguments. */
    _option_value: $ => choice(
      $.option_list,
      $.option_call,
      $.identifier,
      $.signed_number,
      $.string,
    ),

    option_call: $ => prec(2, seq(
      field('func', $.identifier),
      '(',
      optional(field('args', commaSep1(choice(
        $.string, $.signed_number, $.identifier,
      )))),
      ')',
    )),

    option_list: $ => seq(
      '[',
      optional(commaSep1(field('item', choice(
        $.identifier, $.string, $.signed_number,
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
      field('op', choice('>>', '<<')),
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
        bracketedList($, '(', field('args', $.identifier), ')'),
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
      bracketedList($, '(', field('args', $.identifier), ')'),
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
        bracketedList($, '{', field('cases', $.let_factor_case), '}'),
        field('body', $._let_arith),
      ),
    )),

    let_factor_binder: $ => seq(
      field('var', $.identifier),
      ':',
      field('index', $._object_expr),
    ),

    let_factor_case: $ => seq(
      field('label', $.integer),
      '->',
      field('value', $._let_arith),
    ),

    let_list: $ => choice(
      seq('[', ']'),
      bracketedList($, '[', $._let_arith, ']'),
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
      bracketedList($, '[', field('indices', $._let_arith), ']'),
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

    doc_comment_group: $ => repeat1(seq($.doc_comment, $._newline)),

    // -----------------------------------------------------------------
    // tokens
    // -----------------------------------------------------------------

    /* The ``#`` family.
     *
     *   ``# ...``         line comment    -- end-of-line text, no semantics
     *   ``#! ...``        doc comment     -- attaches to next decl as ``docs``
     *   ``#{ ... }#``     block comment   -- multi-line, non-nesting
     *   ``#[ k = v ]``    outer pragma    -- compiler directive on next decl
     *   ``#![ k = v ]``   inner pragma    -- compiler directive on the module
     *
     * The lexer disambiguates by the second character after ``#``:
     * ``!`` opens doc_comment or inner_pragma; ``{`` opens block_comment;
     * ``[`` opens outer_pragma; everything else (including EOL) is a plain
     * line_comment. The pragma forms are STRUCTURAL rules at the parser
     * level, not lexer tokens; the lexer just yields the opening ``#[``
     * or ``#![`` as a literal so the parser can drive the entry list.
     */
    /* Doc comment ``#! ...``. Must NOT match ``#![``, which opens
     * an inner pragma. We require the body to start with a char
     * other than ``[`` (and non-newline); empty-body doc comments
     * (``#!\n``) are intentionally unsupported because they convey
     * nothing and would clash with the pragma opener.
     */
    doc_comment: _ => token(prec(2, seq(
      '#!',
      /[^\[\n][^\n]*/,
    ))),
    block_comment: _ => token(prec(2, seq(
      '#{',
      /[^}]*(?:}[^#][^}]*)*/,
      '}#',
    ))),
    /* Line comments must NOT swallow ``#!``/``#[``/``#![``/``#{``
     * prefixes that introduce one of the richer comment / pragma
     * shapes. The second-character exclusion set is the precise
     * difference. The empty-body case ``#`` followed by EOL is
     * handled via the leading ``choice('', ...)``.
     */
    line_comment: _ => token(seq(
      '#',
      choice('', seq(/[^!\[{\n]/, /[^\n]*/)),
    )),

    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,
    integer: _ => /[0-9]+/,
    /* Floats admit trailing-dot (``1.``), leading-dot (``.5``), and
     * exponent-only (``1e-3``) forms alongside ``1.0`` and
     * ``2.5e-3``. */
    float: _ => /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?|\.[0-9]+([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+/,
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

function commaSep(rule) {
  return optional(commaSep1(rule));
}

/* Bracketed comma-separated list with two forms:
 *
 *   Inline:     ``[ a, b, c ]``      -- single line, no newlines.
 *   Multi-line: ``[\n  a,\n  b,\n]`` -- newline immediately after
 *                                       ``open`` opts into the
 *                                       multi-line form, which then
 *                                       allows newlines (and line
 *                                       / doc / block comments,
 *                                       which are extras) between
 *                                       elements and a trailing
 *                                       comma.
 *
 * The first-token-after-``open`` disambiguates the two forms
 * deterministically: NEWLINE picks multi-line, anything else
 * picks inline.
 *
 * @param {GrammarSymbols<any>} $
 * @param {RuleOrLiteral} open
 * @param {RuleOrLiteral} item
 * @param {RuleOrLiteral} close
 * @returns {ChoiceRule}
 */
function bracketedList($, open, item, close) {
  return choice(
    seq(open, commaSep1(item), optional(','), close),
    seq(
      open,
      $._newline,
      repeat($._newline),
      item,
      repeat(seq(',', repeat1($._newline), item)),
      optional(seq(',', repeat($._newline))),
      close,
    ),
  );
}
