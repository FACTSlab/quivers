; QVR syntax highlighting queries.

; ---------------------------------------------------------------------------
; keywords
; ---------------------------------------------------------------------------

[
  "quantale"
  "category"
  "rule"
  "object"
  "let"
  "output"
  "where"
  "type"
  "space"
  "continuous"
  "stochastic"
  "discretize"
  "embed"
  "program"
  "draw"
  "observe"
  "return"
  "latent"
  "observed"
] @keyword

[
  "identity"
  "fan"
  "repeat"
  "stack"
  "scan"
  "parser"
  "ccg"
  "lambek"
  "marginalize"
] @function.builtin

(let_call func: _ @function.builtin)

; ---------------------------------------------------------------------------
; operators
; ---------------------------------------------------------------------------

[
  "->"
  "=>"
  ">>"
  "<<"
  ">=>"
  "<-"
  "~"
  "@"
  "*"
  "+"
  "/"
  "\\"
  "-"
  "="
  ":"
  "."
] @operator

; ---------------------------------------------------------------------------
; declarations and identifiers
; ---------------------------------------------------------------------------

(quantale_decl name: (identifier) @constant)
(category_decl names: (identifier) @type)
(object_decl   name: (identifier) @type)
(rule_decl     name: (identifier) @function)
(rule_decl     variables: (identifier) @variable.parameter)
(morphism_decl name: (identifier) @function)
(let_decl      name: (identifier) @function)
(continuous_decl name: (identifier) @function)
(stochastic_decl name: (identifier) @function)
(discretize_decl name: (identifier) @function)
(embed_decl    name: (identifier) @function)
(program_decl  name: (identifier) @function)
(space_decl    name: (identifier) @type)
(type_alias_decl name: (identifier) @type)

(space_constructor       constructor: (identifier) @type.builtin)
(space_constructor_bare  constructor: (identifier) @type.builtin)

(continuous_decl family: (identifier) @type)

; identifiers in patterns / expressions
(type_atom   (identifier) @type)
(type_effect_apply effect: (identifier) @type)
(space_atom  (identifier) @type)
(expr_ident  (identifier) @variable)
(let_var     (identifier) @variable)

; ---------------------------------------------------------------------------
; literals
; ---------------------------------------------------------------------------

(integer)       @number
(float)         @number
(signed_number) @number
(line_comment)  @comment
