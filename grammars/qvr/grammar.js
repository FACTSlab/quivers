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
  trans_compose: 1, // >>> (Trans composition)
  compose: 1,    // >> << >=>
  tensor:  2,    // @
  postfix: 3,    // .method(...)
  // type expression precedence:
  type_coproduct: 1,  // +
  type_slash:     2,  // / \   (residuated; binds tighter than +, looser than *)
  type_product:   3,  // *
  type_apply:     4,  // T(X)  effect-typed application
  bind_step:     10,  // outranks type_apply for `(` lookahead at step start
  // let-arithmetic precedence:
  let_add: 1,
  let_mul: 2,
  let_unary: 3,
};

module.exports = grammar({
  name: 'qvr',

  extras: $ => [/\s/, $.doc_comment, $.line_comment],

  word: $ => $.identifier,

  conflicts: $ => [
    // `_let_atom` vs `let_index`: after a `let_var`, a `[` may
    // either continue into a `let_index` (gather expression) or
    // end the atom. tree-sitter cannot decide on one token of
    // lookahead; declaring the conflict triggers GLR exploration
    // and the longer-match wins via let_index's left-precedence.
    [$._let_atom, $.let_index],
  ],

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
      $.contraction_decl,
      $.let_decl,
      $.export_decl,
      $.deduction_decl,
      $.signature_decl,
      $.encoder_decl,
      $.decoder_decl,
      $.loss_decl,
    ),

    // ---------------------------------------------------------------
    // simple declarations
    // ---------------------------------------------------------------

    // Composition-rule declaration.  Four surface forms differ only
    // in the algebraic level they advertise:
    //
    //   ``quantale X``         — X must be a Quantale (full structure;
    //                            identity / dagger / cup / cap are
    //                            available).
    //   ``semigroupoid X``     — X is associative but lacks identity.
    //   ``bilinear_form X``    — X is a CompositionRule with no
    //                            associativity promise.
    //   ``composition_rule X`` — permissive: X is any CompositionRule.
    //
    // Each may optionally carry a user-defined body
    // ``{ tensor_op(a, b) = …; join(t) = …; unit = …; … }`` that
    // declares a fresh rule inline instead of referencing a registered
    // singleton.  The compiler enforces that the body's operations
    // match the keyword's algebraic level.
    quantale_decl: $ => choice(
      seq('quantale',         field('name', $.identifier), optional($.composition_rule_block)),
      seq('semigroupoid',     field('name', $.identifier), optional($.composition_rule_block)),
      seq('bilinear_form',    field('name', $.identifier), optional($.composition_rule_block)),
      seq('composition_rule', field('name', $.identifier), optional($.composition_rule_block)),
    ),

    // Body of a user-defined composition rule.  Each entry is
    // either ``key(p1, p2, …) = body`` (a function-valued field,
    // for tensor_op / join / negation / meet) or ``key = body``
    // (a value-valued field, for unit / zero literals).
    composition_rule_block: $ => seq(
      '{',
      repeat($.composition_rule_entry),
      '}',
    ),

    composition_rule_entry: $ => choice(
      seq(
        field('key', $.identifier),
        '(',
        field('params', commaSep1($.identifier)),
        ')',
        '=',
        field('body', $._let_arith),
      ),
      seq(
        field('key', $.identifier),
        '=',
        field('body', $._let_arith),
      ),
    ),

    // N-ary operadic contraction declaration.  Declares a multi-
    // input morphism whose body is an einsum-style wiring under a
    // named composition rule:
    //
    //   contraction op_apply (
    //       arg1 : A -> B,
    //       arg2 : A -> C,
    //       kernel : (B * C) -> D
    //   ) : A -> D
    //       rule product_fuzzy
    //       wiring "ab, ac, bcd -> ad"
    //
    // The declaration registers a callable named ``op_apply`` that
    // takes the three input morphisms and contracts them under the
    // wiring spec using the rule's tensor_op and join.
    contraction_decl: $ => seq(
      'contraction',
      field('name', $.identifier),
      '(',
      field('inputs', commaSep1($.contraction_input)),
      ')',
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      'rule',
      field('rule_name', $.identifier),
      'wiring',
      field('wiring_spec', $.string),
    ),

    contraction_input: $ => seq(
      field('name', $.identifier),
      ':',
      field('input_domain', $._type_expr),
      '->',
      field('input_codomain', $._type_expr),
    ),

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

    // Weighted-deduction system declaration.
    //
    //   deduction CG : Token -> Cat {
    //       atoms { NP, S, VP }
    //       rule fwd_app  : X/Y, Y       |- X
    //       rule bwd_app  : Y, Y\X       |- X
    //       semiring  ProductFuzzy
    //       start     S
    //       depth     4
    //   }
    //
    // The body is a record of seven canonical parameters of an
    // agenda-based deduction (Shieber-Schabes-Pereira 1995;
    // Goodman 1999): items / atoms, rules (sequent-style),
    // semiring, axiom-source, goal predicate, start symbol,
    // depth bound. Concrete parsing strategies (CKY, Earley, A*,
    // Knuth) are picked by the compiler from these parameters;
    // an explicit `strategy = …` field may override.
    deduction_decl: $ => seq(
      'deduction',
      field('name', $.identifier),
      ':',
      field('domain', $._type_expr),
      '->',
      field('codomain', $._type_expr),
      '{',
      repeat(choice(
        $.deduction_atoms,
        $.deduction_rule,
        $.deduction_semiring,
        $.deduction_start,
        $.deduction_depth,
        $.deduction_lexicon,
        $.deduction_lexicon_from_file,
        $.deduction_axioms,
        $.deduction_signature,
        $.deduction_encoder_attach,
      )),
      '}',
    ),

    deduction_signature: $ => seq(
      'signature',
      field('signature', $.identifier),
    ),

    deduction_encoder_attach: $ => seq(
      'encoder',
      field('encoder', $.identifier),
    ),

    // Atoms block: `atoms { A, B, C }`.
    deduction_atoms: $ => seq(
      'atoms',
      '{',
      field('atoms', commaSep1($.identifier)),
      '}',
    ),

    // Sequent-style rule:
    //   rule name : premise1, premise2, ... |- conclusion
    // Wildcards bind via single-uppercase identifiers; concrete
    // atom names match literally.
    deduction_rule: $ => seq(
      'rule',
      field('name', $.identifier),
      ':',
      field('premises', commaSep1($._type_expr)),
      choice('|-', '⊢'),
      field('conclusion', $._type_expr),
    ),

    deduction_semiring: $ => seq(
      'semiring',
      field('semiring', $.identifier),
    ),

    deduction_start: $ => seq(
      'start',
      field('start', $.identifier),
    ),

    deduction_depth: $ => seq(
      'depth',
      field('depth', $.integer),
    ),

    // Inline lexicon block — for small / hand-built deductions:
    //
    //   lexicon {
    //       "every"  : S/(S\NP) = every_lf  @ learnable
    //       "dog"    : S\NP     = pred_dog  @ learnable
    //   }
    //
    // Each entry maps a literal word string to a (category,
    // LF-template) pair with an optional `@ learnable` modifier
    // that requests a per-entry `nn.Parameter` log-weight.
    deduction_lexicon: $ => seq(
      'lexicon',
      '{',
      repeat($.lexicon_entry),
      '}',
    ),

    lexicon_entry: $ => seq(
      field('word', $.string),
      ':',
      field('category', $._type_expr),
      '=',
      field('lf', $._let_arith),
      optional(seq('@', field('learnable', $.learnable_marker))),
    ),

    learnable_marker: $ => 'learnable',

    // File-loaded lexicon — for large vocabularies:
    //
    //   lexicon from "lexicon.tsv" with learnable
    //
    // The path is resolved relative to the source .qvr file. The
    // file is read at compile time; one nn.Parameter is allocated
    // per row when `with learnable` is present. Supported
    // formats: TSV with three columns `word, category, lf_term`.
    deduction_lexicon_from_file: $ => seq(
      'lexicon',
      'from',
      field('path', $.string),
      optional(seq('with', field('learnable', $.learnable_marker))),
    ),

    // General axiom source — for axioms that are not lexicon-shaped:
    //
    //   axioms = some_morphism_name
    //
    // The named morphism is a kernel `Input -> List[(Item, Weight)]`
    // resolved from the surrounding module's `let` / `continuous` /
    // `stochastic` bindings. `lexicon { … }` and `lexicon from "..."`
    // are sugar for the lexical specialisation of this primitive.
    deduction_axioms: $ => seq(
      'axioms',
      '=',
      field('source', $.identifier),
    ),

    // ---------------------------------------------------------------
    // structural-compression: signatures, encoders, decoders, losses
    // ---------------------------------------------------------------
    //
    // A `signature` block declares the algebra over which encoders
    // and decoders are defined. It contains sorts, constructors (typed
    // operations), binders (operations that introduce scoped variables
    // under a de-Bruijn discipline with explicit bound-variable types),
    // and optionally vertex/edge kinds for graph-shaped signatures.

    signature_decl: $ => seq(
      'signature',
      field('name', $.identifier),
      optional(seq('[', field('params', commaSep1($.identifier)), ']')),
      '{',
      repeat(choice(
        $.signature_sorts,
        $.signature_constructors,
        $.signature_binders,
        $.signature_vertex_kinds,
        $.signature_edge_kinds,
      )),
      '}',
    ),

    signature_sorts: $ => seq(
      'sorts',
      '{',
      repeat(field('sorts', $.sort_decl)),
      '}',
    ),

    sort_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(seq('dim', field('dim', $.integer))),
      optional(seq(
        'vocab',
        '{',
        field('vocab', commaSep1($.vocab_literal)),
        '}',
      )),
      optional(','),
    ),

    sort_kind: $ => choice('object', 'index', 'data'),

    // A closed-vocabulary entry. We accept the three principal
    // data-leaf shapes: string literals, signed integers, and
    // floats. The compiler validates that the host sort is `data`.
    vocab_literal: $ => choice(
      $.string,
      $.integer,
      $.float,
    ),

    signature_constructors: $ => seq(
      'constructors',
      '{',
      repeat(field('constructors', $.constructor_decl)),
      '}',
    ),

    constructor_decl: $ => seq(
      field('name', $.identifier),
      ':',
      optional(field('domain', commaSep1($._sig_sort))),
      '->',
      field('codomain', $._sig_sort),
      optional(','),
    ),

    _sig_sort: $ => prec(1, $.identifier),

    signature_binders: $ => seq(
      'binders',
      '{',
      repeat(field('binders', $.binder_decl)),
      '}',
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
      optional(','),
    ),

    // A binder variable: `var : sort` introduces a variable of the
    // given sort; an optional `: annot : annot_sort` clause attaches
    // a type annotation visible to the encoder / decoder while
    // not itself entering the scope of `body`.
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
      '{',
      repeat(field('vertex_kinds', $.vertex_kind_decl)),
      '}',
    ),

    vertex_kind_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('kind', $.sort_kind),
      optional(seq('dim', field('dim', $.integer))),
      optional(','),
    ),

    signature_edge_kinds: $ => seq(
      'edge_kinds',
      '{',
      repeat(field('edge_kinds', $.edge_kind_decl)),
      '}',
    ),

    edge_kind_decl: $ => seq(
      field('name', $.identifier),
      ':',
      field('src', $.identifier),
      field('arrow', $.edge_arrow),
      field('tgt', $.identifier),
      optional(','),
    ),

    edge_arrow: $ => choice('->', '--'),

    // ---------------------------------------------------------------
    // Encoder declaration: an algebra homomorphism T_Σ -> Vec_D
    // realised by per-constructor parametric functions.

    encoder_decl: $ => seq(
      'encoder',
      field('name', $.identifier),
      'over',
      field('signature', $.identifier),
      optional(seq('[', field('sig_args', commaSep1($.identifier)), ']')),
      '{',
      repeat(choice(
        $.encoder_dim,
        $.encoder_iterations,
        $.encoder_readout,
        $.encoder_op_rule,
        $.encoder_message_rule,
        $.encoder_update_rule,
        $.encoder_init_rule,
        $.encoder_var_init,
      )),
      '}',
    ),

    encoder_dim: $ => seq(
      'dim',
      field('sort', $.identifier),
      '=',
      field('dim', $.integer),
    ),

    encoder_iterations: $ => seq(
      'iterations',
      field('iterations', $.integer),
    ),

    encoder_readout: $ => seq(
      'readout',
      '|->',
      field('body', $._let_arith),
    ),

    // Per-constructor rule. The mode controls how arguments are
    // threaded:
    //   <constructor>(arg1, ..., argN)              |-> body   (plain)
    //   <constructor>(...)  recurrent <state>       |-> body   (sequence)
    //   <constructor>(...)  attention <prefix>      |-> body   (transformer)
    encoder_op_rule: $ => seq(
      field('op', $.identifier),
      optional(seq('(', commaSep1(field('args', $.identifier)), ')')),
      optional(choice(
        seq('recurrent', field('state', $.identifier)),
        seq('attention', field('prefix', $.identifier)),
      )),
      '|->',
      field('body', $._let_arith),
    ),

    encoder_init_rule: $ => seq(
      'init',
      field('kind', $.identifier),
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
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
    ),

    // Per-(var_sort, annot_sort) `var_init` body. Multiple
    // declarations per encoder are permitted, one per pair the
    // signature's binders introduce. The `from <annot_sort>` clause
    // is omitted for unannotated binders.
    //
    //   var_init Term from Type as ty   |-> mlp_tv(ty)
    //   var_init Type                   |-> type_var_init
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
    ),

    // ---------------------------------------------------------------
    // Decoder declaration: a Kleisli arrow Vec_D -> Kern(T_Σ).

    decoder_decl: $ => seq(
      'decoder',
      field('name', $.identifier),
      'over',
      field('signature', $.identifier),
      optional(seq('[', field('sig_args', commaSep1($.identifier)), ']')),
      optional(seq('depth', field('depth', $.integer))),
      '{',
      repeat(choice(
        $.decoder_dim,
        $.decoder_structure,
        $.decoder_primitive,
        $.decoder_factor,
        $.decoder_binder_select,
        $.decoder_body_default,
      )),
      '}',
    ),

    decoder_dim: $ => seq(
      'dim',
      field('sort', $.identifier),
      '=',
      field('dim', $.integer),
    ),

    decoder_structure: $ => seq(
      'structure',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
    ),

    decoder_primitive: $ => seq(
      'primitive',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
    ),

    decoder_factor: $ => seq(
      'factor',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
    ),

    decoder_binder_select: $ => seq(
      'binder_select',
      '(',
      field('arg', $.identifier),
      ')',
      '|->',
      field('body', $._let_arith),
    ),

    decoder_body_default: $ => seq(
      'body',
      '|->',
      field('default', 'recursive'),
    ),

    // ---------------------------------------------------------------
    // Loss declaration: attachable, weighted scalar objectives.

    loss_decl: $ => seq(
      'loss',
      field('name', $.identifier),
      optional(seq('weight', field('weight', $._let_arith))),
      optional(seq('on', field('attachment', $.loss_attachment))),
      '{',
      field('body', $._let_arith),
      '}',
    ),

    loss_attachment: $ => choice(
      seq(field('kind', $.loss_attachment_kind),
          field('target', $.identifier)),
      seq('rule', field('rule_name', $.identifier), 'in',
          field('deduction', $.identifier)),
      seq('chart', 'of', field('chart_of', $.identifier)),
    ),

    loss_attachment_kind: $ => choice(
      'program', 'deduction', 'encoder', 'decoder',
    ),

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
      $.trans_compose,
      $.compose_expr,
      $.tensor_expr,
      $.postfix_expr,
      $._atom_expr,
    ),

    // Transformation composition.  ``t1 >>> t2`` denotes the
    // sequential application of two :class:`MorphismTransformation`
    // (or :class:`QuantaleHomomorphism`) values.  Distinct from
    // ``>>`` (V-Cat morphism composition): ``>>`` composes
    // morphisms within a quantale; ``>>>`` composes the change-of-
    // base transformations between quantales.  Required type:
    // ``t1.target == t2.source`` (checked at compile time).
    trans_compose: $ => prec.left(PREC.trans_compose, seq(
      field('left',  $._expr),
      '>>>',
      field('right', $._expr),
    )),

    // Composition operators. Each one carries its enrichment
    // quantale so the V-Cat composition dispatches to that
    // quantale's monoidal structure regardless of the operands'
    // declared quantale. The operator set was chosen to share
    // family resemblance with canonical operators in other
    // languages rather than clashing with them:
    //
    //   >>   ProductFuzzy noisy-OR (the default; family-resembles
    //        Haskell's ``>>`` for Kleisli sequencing).
    //   <<   Reverse ``>>`` (Haskell ``<<``-shaped).
    //   >=>  Kleisli composition (Haskell's ``>=>`` — direct
    //        family match).
    //   *>   Markov sum-product (family-resembles Haskell's
    //        Applicative ``*>``: both sequence two operations
    //        in a single arrow).
    //   ~>   LogProb sum-product in log-space (family-resembles
    //        the natural-transformation ``~>`` used in Haskell
    //        / lens libraries).
    //   ||>  Gödel (min / max with Heyting implication). The
    //        ``||`` shape echoes the logical-OR symbol; Gödel's
    //        join is max which is the fuzzy extension of OR.
    //   ?>   Viterbi (max-plus tropical, best path). The ``?``
    //        reads as "which choice is best" — Viterbi is the
    //        MAP-decoding semiring.
    //   &&>  Boolean (∧ / ∨). The ``&&`` shape echoes the
    //        logical-AND symbol — Boolean's tensor is AND.
    //   +>   Łukasiewicz (probabilistic sum bounded by 1). The
    //        ``+`` evokes the "soft OR" sum operation of the
    //        Łukasiewicz t-conorm.
    //   $>   Real sum-product on ℝ (canonical numeric semiring;
    //        ⊕ = +, ⊗ = ·). The ``$`` evokes "real value".
    //   %>   Probability sum-product on [0, 1] (same operations
    //        as $>, clamped to the unit interval). The ``%``
    //        evokes "percentage".
    //
    // Cross-operator composition (mixing ``>>`` and ``*>`` in a
    // single chain) requires an explicit ``.change_base(φ)``
    // between the two segments — the operator carries the
    // quantale but does not auto-convert operands.
    compose_expr: $ => prec.left(PREC.compose, seq(
      field('left',  $._expr),
      field('op',    choice(
        '>>', '<<', '>=>',
        '*>', '~>', '||>', '?>', '&&>', '+>',
        '$>', '%>',
      )),
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
      // residuation-witness combinators: given f : X * Y -> Z
      // where Z lives in a residuated universe, produce f.curry_right :
      // X -> Z/Y or f.curry_left : Y -> X\Z. No arguments.
      seq(field('name', choice('curry_right', 'curry_left'))),
      // change-of-base: given f : A -> B over quantale V and a
      // transformation φ : Trans[V, W] (a QuantaleHomomorphism
      // or MorphismTransformation), ``f.change_base(phi)`` is
      // the V-Cat morphism A -> B over W with tensor
      // φ.apply(f.tensor).  The argument is any expression that
      // evaluates to a Trans value: a bare name (registered
      // singleton or let-bound), a constructor call
      // ``softmax(B)`` / ``bayes_invert(prior)``, or a
      // composition ``t1 >>> t2``.
      seq(
        field('name', 'change_base'),
        '(',
        field('arg', $._expr),
        ')',
      ),
      // compact-closed surface: ``f.dagger`` for the transpose,
      // ``f.trace(A)`` for the trace along object A.
      seq(field('name', 'dagger')),
      seq(
        field('name', 'trace'),
        '(',
        field('args', $.identifier),
        ')',
      ),
      // freeze: materialise an expression as a frozen
      // :class:`ObservedMorphism`. The resulting morphism's
      // parameters do not propagate gradients to the constituent
      // morphisms; gradient flow stops at the freeze. Used to
      // pin a learned composition as a structural input to a
      // downstream model — equivalent to detach() on the tensor.
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

    // Call expression for n-ary categorical operations declared
    // by a ``contraction`` block.  Surface form ``name(arg1, arg2,
    // ...)`` where ``name`` resolves a registered contraction and
    // each argument is the name of a morphism in scope.  Higher
    // precedence than bare identifier so ``foo(bar)`` parses as a
    // call rather than ``foo`` followed by ``(bar)``.
    morphism_call: $ => prec(20, seq(
      field('callee', $.identifier),
      '(',
      field('args', commaSep1($.identifier)),
      ')',
    )),

    expr_paren: $ => seq('(', $._expr, ')'),

    expr_ident: $ => $.identifier,

    identity_expr: $ => seq(
      'identity', '(', field('object', $.identifier), ')',
    ),

    // Compact-closed unit / counit. ``cup(A)`` builds the
    // morphism ``I -> A * A`` whose tensor is the diagonal on A
    // (every entry ``(a, a)`` carries the quantale's monoidal
    // unit). ``cap(A)`` is the dual ``A * A -> I``. Together
    // ``cup`` and ``cap`` provide the unit / counit of the
    // compact-closed structure on V-Cat; the snake equations
    // ``(ε ⊗ id) ∘ (id ⊗ η) = id`` hold by construction.
    cup_expr: $ => seq(
      'cup', '(', field('object', $.identifier), ')',
    ),

    cap_expr: $ => seq(
      'cap', '(', field('object', $.identifier), ')',
    ),

    // Data-derived initialiser: ``from_data("KEY")`` resolves the
    // string literal as a key into the runtime-supplied data
    // dictionary at fit time, and the morphism's tensor is the
    // looked-up value. The result is an :class:`ObservedMorphism`
    // — the entries are structural / frozen, not learnable. Used
    // for embeddings loaded from a file, adjacency matrices,
    // dataset-derived priors, fixed parse structures.
    from_data_expr: $ => seq(
      'from_data',
      '(',
      field('key', $._string_literal),
      ')',
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
    bind_step: $ => prec.dynamic(PREC.bind_step, prec.right(seq(
      field('vars', $._var_pattern),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    ))),

    // Scored bind — same shape as `bind_step` but prefixed with
    // `observe`, marking the bound coordinate as clamped at runtime
    // by the `observations` dict; the resulting kernel becomes
    // sub-probabilistic.
    //
    //   observe v        <- F(args)
    //   observe r : N    <- F(theta[N])   -- N-indexed batched score
    //   observe r : N via idx <- F(...)   -- per-observe fibration
    //   observe r : N via product(a, b) <- F(...)
    //
    // Inside a grouped `marginalize` block (header carries
    // ``over G`` or ``over G * H``), every observe step MUST
    // carry its own ``via <idx>`` (or ``via product(...)``)
    // clause.  The compiler scatter-adds each observe's per-row
    // per-class log-likelihood into the same per-group
    // accumulator before the reduction.  Outside a grouped body
    // ``via`` on an observe is a compile-time error.
    observe_step: $ => prec.right(seq(
      'observe',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      optional(seq('via', field('via', $._via_spec))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
    )),

    // Scoped marginalisation. Introduces a coordinate `c` bound to a
    // kernel `F(args)`, optionally indexed by `: A`; the body in `{ … }`
    // is the integration scope. At the end of the scope the coordinate
    // is pushed forward through the projection (logsumexp for discrete,
    // fibrewise integration for continuous), and `c` falls out of
    // scope.
    //
    // A grouped block additionally declares a grouping plate
    // ``over G`` (or a product plate ``over G * H``).  Inside the
    // body, every observe step carries its own ``via <idx>``
    // clause naming the per-observe fibration into the shared
    // grouping plate.  The compiler scatter-adds each observe's
    // per-row per-class log-likelihood into the same
    // ``(|G|, K)`` accumulator before the reduction:
    //
    //     Σ_g logsumexp_k [ log π(g, k) +
    //                       Σ_m Σ_{n: idx_m(n)=g} ℓ_m(n, k) ]
    //
    // realising the right Kan extension along the coproduct
    // fibration ⨿_m r_m in Kern.  The single-observe case is the
    // unary slice (M = 1).
    //
    //   marginalize class : K <- Categorical(probs) in {
    //       observe r : N <- Bernoulli(theta[class[N]])
    //   }
    //
    //   marginalize class : K <- Categorical(probs)
    //       over G
    //       in {
    //           let logit = base + sign[class]
    //           observe r : N via idx <- Bernoulli(logit)
    //       }
    marginalize_step: $ => seq(
      'marginalize',
      field('var', $.identifier),
      optional(seq(':', field('index', $._type_expr))),
      '<-',
      field('morphism', $.identifier),
      optional(seq('(', field('args', commaSep1($._draw_arg)), ')')),
      // `over G` declares a single grouping plate; `over G * H`
      // declares a product grouping plate whose flat cardinality is
      // |G|·|H|. The compiler resolves the type-product into a
      // tuple of plate cardinalities and pairs it with the
      // co-indexed `via` fibrations declared on each observe in
      // the body.
      optional(seq('over', field('over', $._type_expr))),
      // `reduction = logsumexp | sum | mean` controls the per-group
      // reduction over the class axis: `logsumexp` is the canonical
      // mixture-marginalisation form, `sum` is the joint scoring
      // form (used by predictive paths), `mean` is the symmetric
      // average. Defaults to `logsumexp`.
      optional(seq('reduction', '=', field('reduction', $.identifier))),
      'in',
      '{',
      field('scope', repeat($._program_step)),
      '}',
    ),

    // Fibration specification: either a single identifier or a
    // `product(...)` of identifiers naming the per-axis fibrations.
    _via_spec: $ => choice(
      $.identifier,
      $.via_product,
    ),

    via_product: $ => seq(
      'product',
      '(',
      commaSep1(field('axis', $.identifier)),
      ')',
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

    // Destructuring tuple bind uses square brackets to disambiguate
    // from a `(...)` opening a `type_effect_apply` continuation of
    // the program's codomain type. Parens-prefixed tuple binds
    // would create an unresolvable LR(1) ambiguity at the
    // boundary between the codomain and the first program step.
    //
    //   [a, b] <- F(args)        -- destructure F's tuple return
    //   [a, b, c] <- sub(...)    -- destructure a sub-program
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

    // Labelled-tuple return: `return (a: x, b: y)`. Renames the
    // coordinates of the resulting product space — purely
    // syntactic rebinding at the schema level; preserves the
    // categorical denotation up to coordinate renaming.
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
      $.let_method_call,
      $.let_call,
      $.let_index,
      $.let_list,
      $.let_lambda,
      $.let_string,
      $.let_var,
      $.let_literal,
    ),

    // List literal in let-expressions: `[a, b, c]`. Categorically
    // an element of the free monoid `let_arith^*` over the
    // arithmetic / let-value sublanguage; the runtime represents
    // it as a Python list with autograd-transparent contents.
    let_list: $ => seq(
      '[',
      optional(seq(
        commaSep1($._let_arith),
        optional(','),
      )),
      ']',
    ),

    // String literal: `"foo"`. Used for tokenisation, lexicon
    // keys, and as ground-atom names in LF constructors like
    // `pred("dog")`.
    let_string: $ => $._string_literal,

    // Lambda expression: `param -> body`. Categorically a curried
    // function in the Kleisli setting; the runtime evaluator
    // closes over the surrounding let environment when
    // instantiating the closure.
    let_lambda: $ => prec.right(seq(
      field('param', $.identifier),
      '->',
      field('body', $._let_arith),
    )),

    // Method-call expression: `receiver.method(args)`. Used to
    // dispatch chart queries (`chart.weight(item)`,
    // `chart.enumerate(pattern)`, `chart.goal_weight()`) and any
    // future ChartView API. The receiver is always a let_var so
    // the runtime can resolve it from the environment.
    let_method_call: $ => prec.left(2, seq(
      field('receiver', $.let_var),
      '.',
      field('method', $.identifier),
      '(',
      optional(field('args', commaSep1($._let_arith))),
      ')',
    )),

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

    // Generalised let-call: `func(args, ...)`.
    //
    // The function name is any identifier. The runtime
    // dispatch handles three cases:
    //
    //   1. Built-in numeric morphisms (`sigmoid`, `exp`, `log`,
    //      `abs`, `softplus`, `cumsum`, `softmax`,
    //      `cholesky_quad_form`) — evaluated as the corresponding
    //      torch operations on tensor inputs.
    //
    //   2. Built-in higher-order combinators (`logsumexp_over`,
    //      `fold`, `map`, `filter`, `length`, `parse`) — evaluated
    //      with their declarative semantics; lambdas in arg
    //      position are closures over the local environment.
    //
    //   3. *Constructor* application (anything else): produces a
    //      structured tuple `(func_name, *args)`. This is the
    //      LF-construction mode — `pred("dog")` builds
    //      `("pred", "dog")`; `forall("x", body)` builds
    //      `("forall", "x", body)`; `implies(p, q)` builds
    //      `("implies", p, q)`. The runtime treats these tuples
    //      as ordinary chart items.
    //
    // Categorically, the constructor mode realises the free
    // term algebra over the named operation symbols, embedding
    // it as values in the let-sublanguage.
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

    // String literals: double-quoted, with backslash escapes.
    // The grammar restricts to single-line strings (no embedded
    // newlines); multiline strings are not part of the v0.5
    // surface.
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
