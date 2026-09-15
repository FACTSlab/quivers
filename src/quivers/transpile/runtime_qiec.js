var _qvr_qiec_pure = function(value) { return { tag: "pure", value: value }; };
// A deferred computation. Calls are forced by the trampolines in _qvr_qiec_run and
// the handler walk rather than when they are built, so a recursive QIEC computation
// does not consume host stack per call. `frames` are the dynamic address frames the
// result runs under, and `tail` records that the call retires the frame of the
// computation that made it.
var _qvr_qiec_call = function(thunk, frames, tail) { return { tag: "call", thunk: thunk, frames: frames || [], tail: tail === true }; };
var _qvr_qiec_join_frames = function(outer, inner, tail) {
  if (tail && outer.length && outer[outer.length - 1][0] === "call") { return outer.slice(0, -1).concat(inner); }
  return outer.concat(inner);
};
var _qvr_qiec_scoped = function(comp, frames) {
  if (!frames.length || comp.tag === "pure") { return comp; }
  if (comp.tag === "call") { return { tag: "call", thunk: comp.thunk, frames: _qvr_qiec_join_frames(frames, comp.frames, comp.tail), tail: false }; }
  if (comp.tag === "bind") {
    return { tag: "bind", inner: _qvr_qiec_scoped(comp.inner, frames), continuation: function(value) { return _qvr_qiec_scoped(comp.continuation(value), frames); }, captures: comp.captures };
  }
  var request = Object.assign({}, comp.request);
  request.address = [request.address[0], frames.concat(request.address[1]), request.address[2]];
  return { tag: "effect", request: request, continuation: function(value) { return _qvr_qiec_scoped(comp.continuation(value), frames); } };
};
// The trampoline. Deferred calls are forced and deferred binds are unfolded onto an
// explicit continuation stack, so however deep a recursion is, the host stack stays
// flat. A request surfacing beneath pending binds carries them in its continuation,
// and their captures, so a multi-shot resumption still sees everything it copies.
var _qvr_qiec_force = function(comp, pending) {
  var stack = (pending || []).slice();
  var current = comp;
  while (true) {
    if (current.tag === "call") { current = _qvr_qiec_scoped(current.thunk(), current.frames); }
    else if (current.tag === "bind") { stack.push([current.continuation, current.captures]); current = current.inner; }
    else if (current.tag === "pure") {
      if (!stack.length) { return current; }
      current = stack.pop()[0](current.value);
    } else {
      if (!stack.length) { return current; }
      var request = Object.assign({}, current.request);
      var captures = request.captures || [];
      stack.forEach(function(entry) { captures = captures.concat(entry[1]); });
      request.captures = captures;
      var rest = stack.slice();
      var resumeEffect = current.continuation;
      return { tag: "effect", request: request, continuation: function(value) { return _qvr_qiec_force(resumeEffect(value), rest); } };
    }
  }
};
var _qvr_qiec_serials = { call: 0, instance: 0 };
var _qvr_qiec_enter_call = function(name, thunk, tail) {
  _qvr_qiec_serials.call += 1;
  return _qvr_qiec_call(thunk, [["call", name + "#" + _qvr_qiec_serials.call]], tail);
};
var _qvr_qiec_instance = function(thunk) {
  _qvr_qiec_serials.instance += 1;
  return _qvr_qiec_call(thunk, [["instance", _qvr_qiec_serials.instance]], false);
};
var _qvr_qiec_resume = function(resume, value) { return _qvr_qiec_as_computation(resume(value)); };
var _qvr_qiec_if = function(condition, then, otherwise) {
  condition = _qvr_qiec_value(condition);
  if (typeof condition !== "boolean") { throw new Error("QIEC if condition is not a Boolean"); }
  return condition ? then() : otherwise();
};
var _qvr_qiec_div_int = function(a, b) {
  if (b === 0) { throw new Error("QIEC integer division by zero"); }
  return Math.trunc(a / b);
};
// The closed primitive table. Names and semantics mirror the kernel registry;
// integer division and remainder truncate toward zero on every host.
var _qvr_qiec_primitives = {
  add_int: function(a, b) { return a + b; },
  sub_int: function(a, b) { return a - b; },
  mul_int: function(a, b) { return a * b; },
  div_int: _qvr_qiec_div_int,
  mod_int: function(a, b) { return a - b * _qvr_qiec_div_int(a, b); },
  neg_int: function(a) { return -a; },
  abs_int: function(a) { return Math.abs(a); },
  min_int: function(a, b) { return Math.min(a, b); },
  max_int: function(a, b) { return Math.max(a, b); },
  add_real: function(a, b) { return a + b; },
  sub_real: function(a, b) { return a - b; },
  mul_real: function(a, b) { return a * b; },
  div_real: function(a, b) { return a / b; },
  neg_real: function(a) { return -a; },
  abs_real: function(a) { return Math.abs(a); },
  min_real: function(a, b) { return Math.min(a, b); },
  max_real: function(a, b) { return Math.max(a, b); },
  pow_real: function(a, b) { return Math.pow(a, b); },
  exp: function(a) { return Math.exp(a); },
  log: function(a) { return Math.log(a); },
  sqrt: function(a) { return Math.sqrt(a); },
  eq_int: function(a, b) { return a === b; },
  ne_int: function(a, b) { return a !== b; },
  lt_int: function(a, b) { return a < b; },
  le_int: function(a, b) { return a <= b; },
  gt_int: function(a, b) { return a > b; },
  ge_int: function(a, b) { return a >= b; },
  eq_real: function(a, b) { return a === b; },
  ne_real: function(a, b) { return a !== b; },
  lt_real: function(a, b) { return a < b; },
  le_real: function(a, b) { return a <= b; },
  gt_real: function(a, b) { return a > b; },
  ge_real: function(a, b) { return a >= b; },
  eq_bool: function(a, b) { return a === b; },
  ne_bool: function(a, b) { return a !== b; },
  eq_string: function(a, b) { return a === b; },
  ne_string: function(a, b) { return a !== b; },
  and: function(a, b) { return a && b; },
  or: function(a, b) { return a || b; },
  not: function(a) { return !a; },
  concat: function(a, b) { return a + b; },
  int_to_real: function(a) { return a; },
  int_to_string: function(a) { return String(a); },
  real_to_int: function(a) { return Math.trunc(a); },
  expm1: function(a) { return Math.expm1(a); },
  log1p: function(a) { return Math.log1p(a); },
  log2: function(a) { return Math.log2(a); },
  log10: function(a) { return Math.log10(a); },
  rsqrt: function(a) { return 1 / Math.sqrt(a); },
  square: function(a) { return a * a; },
  sign: function(a) { return Math.sign(a); },
  reciprocal: function(a) { return 1 / a; },
  sin: function(a) { return Math.sin(a); },
  cos: function(a) { return Math.cos(a); },
  tan: function(a) { return Math.tan(a); },
  asin: function(a) { return Math.asin(a); },
  acos: function(a) { return Math.acos(a); },
  atan: function(a) { return Math.atan(a); },
  sinh: function(a) { return Math.sinh(a); },
  cosh: function(a) { return Math.cosh(a); },
  tanh: function(a) { return Math.tanh(a); },
  asinh: function(a) { return Math.asinh(a); },
  acosh: function(a) { return Math.acosh(a); },
  atanh: function(a) { return Math.atanh(a); },
  floor: function(a) { return Math.floor(a); },
  ceil: function(a) { return Math.ceil(a); },
  round: function(a) { return Math.round(a); },
  trunc: function(a) { return Math.trunc(a); },
  erf: function(a) { return _qvr_qiec_erf(a); },
  erfc: function(a) { return 1 - _qvr_qiec_erf(a); },
  erfinv: function(a) { return _qvr_qiec_erfinv(a); },
  lgamma: function(a) { return _qvr_qiec_lgamma(a); },
  digamma: function(a) { return _qvr_qiec_digamma(a); },
  sigmoid: function(a) { return _qvr_qiec_sigmoid(a); },
  relu: function(a) { return Math.max(a, 0); },
  relu6: function(a) { return Math.min(Math.max(a, 0), 6); },
  elu: function(a) { return a > 0 ? a : Math.expm1(a); },
  selu: function(a) { return 1.0507009873554805 * (a > 0 ? a : 1.6732632423543772 * (Math.exp(a) - 1)); },
  gelu: function(a) { return 0.5 * a * (1 + _qvr_qiec_erf(a / Math.sqrt(2))); },
  silu: function(a) { return a * _qvr_qiec_sigmoid(a); },
  mish: function(a) { return a * Math.tanh(_qvr_qiec_softplus(a)); },
  softplus: function(a) { return _qvr_qiec_softplus(a); },
  logsigmoid: function(a) { return -_qvr_qiec_softplus(-a); },
  softsign: function(a) { return a / (1 + Math.abs(a)); },
  as_weight: function(a) { return a; },
  weight_value: function(a) { return a; },
  add_weight: function(a, b) { return a + b; },
  scale_weight: function(a, b) { return a * b; }
};
var _qvr_qiec_sigmoid = function(value) {
  if (value >= 0) { return 1 / (1 + Math.exp(-value)); }
  var exponent = Math.exp(value);
  return exponent / (1 + exponent);
};
var _qvr_qiec_softplus = function(value) { return Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value))); };
// The error function by its Taylor series near the origin and by the
// continued fraction of the complementary function in the tails.
var _qvr_qiec_erf = function(value) {
  var sign = value < 0 ? -1 : 1;
  var x = Math.abs(value);
  if (x < 2.5) {
    var term = x, total = x, n = 0;
    while (Math.abs(term) > 1e-17 * Math.abs(total) && n < 200) {
      n += 1;
      term *= -x * x / n;
      total += term / (2 * n + 1);
    }
    return sign * 2 / Math.sqrt(Math.PI) * total;
  }
  var fraction = x;
  for (var k = 60; k >= 1; k--) { fraction = x + (k / 2) / fraction; }
  return sign * (1 - Math.exp(-x * x) / Math.sqrt(Math.PI) / fraction);
};
var _qvr_qiec_erfinv = function(value) {
  if (value < -1 || value > 1) { throw new Error("erfinv is defined on [-1, 1]"); }
  if (value === 1) { return Infinity; }
  if (value === -1) { return -Infinity; }
  if (value === 0) { return 0; }
  var sign = value > 0 ? 1 : -1;
  var magnitude = Math.abs(value);
  var estimate;
  if (magnitude < 0.7) {
    var square = magnitude * magnitude;
    estimate = magnitude * (((-0.140543331 * square + 0.914624893) * square - 1.645349621) * square + 0.886226899) /
      ((((0.012229801 * square - 0.329097515) * square + 1.442710462) * square - 2.118377725) * square + 1);
  } else {
    var tail = Math.sqrt(-Math.log((1 - magnitude) / 2));
    estimate = (((1.641345311 * tail + 3.429567803) * tail - 1.62490649) * tail - 1.970840454) / ((1.637067800 * tail + 3.543889200) * tail + 1);
  }
  for (var i = 0; i < 3; i++) { estimate -= (_qvr_qiec_erf(estimate) - magnitude) / (2 / Math.sqrt(Math.PI) * Math.exp(-estimate * estimate)); }
  return sign * estimate;
};
// Lanczos approximation of log gamma.
var _qvr_qiec_lgamma = function(value) {
  if (value < 0.5) { return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * value))) - _qvr_qiec_lgamma(1 - value); }
  var coefficients = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  var x = value - 1;
  var total = 0.99999999999980993;
  for (var i = 0; i < coefficients.length; i++) { total += coefficients[i] / (x + i + 1); }
  var t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(total);
};
var _qvr_qiec_digamma = function(value) {
  if (value <= 0 && value === Math.floor(value)) { throw new Error("digamma has a pole at nonpositive integers"); }
  if (value < 0) { return _qvr_qiec_digamma(1 - value) - Math.PI / Math.tan(Math.PI * value); }
  var result = 0;
  while (value < 6) { result -= 1 / value; value += 1; }
  var inverse = 1 / value;
  var square = inverse * inverse;
  return result + Math.log(value) - 0.5 * inverse - square * (1 / 12 - square * (1 / 120 - square * (1 / 252 - square * (1 / 240 - square / 132))));
};
var _qvr_qiec_broadcast = function(implementation, args) {
  var tensors = args.filter(function(argument) { return Array.isArray(argument); });
  if (!tensors.length) { return implementation.apply(null, args); }
  var length = tensors[0].length;
  if (tensors.some(function(tensor) { return tensor.length !== length; })) { throw new Error("QIEC primitive applied to tensors of differing shapes"); }
  var out = [];
  for (var index = 0; index < length; index++) {
    out.push(_qvr_qiec_broadcast(implementation, args.map(function(argument) { return Array.isArray(argument) ? argument[index] : argument; })));
  }
  return Object.freeze(out);
};
var _qvr_qiec_primitive = function(name, args) {
  if (!Object.prototype.hasOwnProperty.call(_qvr_qiec_primitives, name)) { throw new Error("unknown QIEC primitive " + name); }
  return _qvr_qiec_broadcast(_qvr_qiec_primitives[name], args.map(_qvr_qiec_value));
};
var _qvr_qiec_gather = function(value, index) {
  value = _qvr_qiec_value(value);
  index = _qvr_qiec_value(index);
  if (!Array.isArray(value)) { throw new Error("QIEC gather from a non-tensor runtime value"); }
  if (Array.isArray(index)) { return Object.freeze(index.map(function(item) { return _qvr_qiec_gather(value, item); })); }
  return value[index];
};
var _qvr_qiec_flat = function(value) {
  if (!Array.isArray(value)) { return [value]; }
  var out = [];
  value.forEach(function(item) { out = out.concat(_qvr_qiec_flat(item)); });
  return out;
};
var _qvr_qiec_reduce = function(operator, value) {
  var entries = _qvr_qiec_flat(_qvr_qiec_value(value));
  var total = entries.reduce(function(a, b) { return a + b; }, 0);
  if (operator === "sum") { return total; }
  if (operator === "mean") { return total / entries.length; }
  if (operator === "max") { return Math.max.apply(null, entries); }
  if (operator === "min") { return Math.min.apply(null, entries); }
  if (operator === "prod") { return entries.reduce(function(a, b) { return a * b; }, 1); }
  var peak = Math.max.apply(null, entries);
  return peak + Math.log(entries.reduce(function(a, b) { return a + Math.exp(b - peak); }, 0));
};
var _qvr_qiec_rowwise = function(operator, value) {
  value = _qvr_qiec_value(value);
  if (value.length && Array.isArray(value[0])) { return Object.freeze(value.map(function(item) { return _qvr_qiec_rowwise(operator, item); })); }
  var row = value.map(Number);
  var total;
  if (operator === "softmax") {
    var peak = Math.max.apply(null, row);
    var weights = row.map(function(item) { return Math.exp(item - peak); });
    total = weights.reduce(function(a, b) { return a + b; }, 0);
    return Object.freeze(weights.map(function(weight) { return weight / total; }));
  }
  if (operator === "log_softmax") {
    var top = Math.max.apply(null, row);
    var normalizer = top + Math.log(row.reduce(function(a, b) { return a + Math.exp(b - top); }, 0));
    return Object.freeze(row.map(function(item) { return item - normalizer; }));
  }
  if (operator === "cumsum") {
    var running = 0;
    return Object.freeze(row.map(function(item) { running += item; return running; }));
  }
  if (operator === "sort") { return Object.freeze(row.slice().sort(function(a, b) { return a - b; })); }
  total = row.reduce(function(a, b) { return a + b; }, 0);
  return Object.freeze(row.map(function(item) { return item / total; }));
};
var _qvr_qiec_weight_sum = function(value) { return _qvr_qiec_flat(_qvr_qiec_value(value)).reduce(function(a, b) { return a + b; }, 0); };
var _qvr_qiec_segment_sum = function(value, index, groups) {
  var totals = [];
  for (var g = 0; g < groups; g++) { totals.push(0); }
  var weights = _qvr_qiec_value(value);
  var members = _qvr_qiec_value(index);
  for (var i = 0; i < weights.length; i++) { totals[members[i]] += weights[i]; }
  return Object.freeze(totals);
};
var _qvr_qiec_project = function(value, position) {
  value = _qvr_qiec_value(value);
  if (!Array.isArray(value) || position >= value.length) { throw new Error("QIEC projection from a non-product runtime value"); }
  return value[position];
};
var _qvr_qiec_authored = {};
var _qvr_qiec_static_kind = function(argument) {
  var kind = argument && argument.kind;
  if (["type-variable", "type-application", "function-type", "equality-type"].indexOf(kind) >= 0) { return "type"; }
  if (["effect-variable", "effect-ref"].indexOf(kind) >= 0) { return "effect"; }
  return "index";
};
var _qvr_qiec_static_environment = function(telescope, args) {
  args = args || [];
  var ordered;
  if (Array.isArray(args)) { ordered = args; }
  else {
    var expected = telescope.map(function(binder) { return binder.name; }).sort();
    if (JSON.stringify(Object.keys(args).sort()) !== JSON.stringify(expected)) { throw new Error("QIEC static arguments do not match the checked telescope"); }
    ordered = telescope.map(function(binder) { return args[binder.name]; });
  }
  if (ordered.length !== telescope.length) { throw new Error("QIEC static argument arity does not match the checked telescope"); }
  var environment = Object.create(null);
  telescope.forEach(function(binder, index) {
    if (_qvr_qiec_static_kind(ordered[index]) !== binder.kind) { throw new Error("QIEC static argument kind does not match the checked telescope"); }
    environment[binder.name] = ordered[index];
  });
  return environment;
};
var _qvr_qiec_specialize = function(value, environment) {
  if (Array.isArray(value)) { return value.map(function(item) { return _qvr_qiec_specialize(item, environment); }); }
  if (!value || typeof value !== "object") { return value; }
  if (["type-variable", "index-variable", "effect-variable"].indexOf(value.kind) >= 0) {
    var staticKey = value.identity || value.name;
    if (Object.prototype.hasOwnProperty.call(environment, staticKey)) { return environment[staticKey]; }
  }
  var result = {};
  Object.keys(value).forEach(function(key) { result[key] = _qvr_qiec_specialize(value[key], environment); });
  return result;
};
var _qvr_qiec_ambient_handlers = [];
var _qvr_qiec_unique = function(items) {
  return items.filter(function(item, index) { return items.indexOf(item) === index; });
};
var _qvr_qiec_with_ambient = function(controller, thunk) {
  _qvr_qiec_ambient_handlers.push(controller);
  try { return thunk(); }
  finally { _qvr_qiec_ambient_handlers.pop(); }
};
var _qvr_qiec_effect = function(request, staticEnvironment) {
  request = _qvr_qiec_specialize(request, staticEnvironment);
  request = Object.assign({ captures: [] }, request);
  request.handler_captures = _qvr_qiec_unique((request.handler_captures || []).concat(_qvr_qiec_ambient_handlers));
  request.handler_states = request.handler_captures.map(function(controller) { return controller.capture(); });
  return { tag: "effect", request: request, continuation: _qvr_qiec_pure };
};
var _qvr_qiec_bind = function(comp, continuation, captures) {
  captures = captures || [];
  if (comp.tag === "pure") { return continuation(comp.value); }
  if (comp.tag === "call" || comp.tag === "bind") { return { tag: "bind", inner: comp, continuation: continuation, captures: captures }; }
  var request = Object.assign({}, comp.request);
  request.captures = (request.captures || []).concat(captures);
  return { tag: "effect", request: request, continuation: function(value) {
    return _qvr_qiec_bind(comp.continuation(value), continuation, captures);
  } };
};
var _qvr_qiec_is_binding = function(value) {
  return value && value.qiec === "binding" && Object.prototype.hasOwnProperty.call(value, "value");
};
var _qvr_qiec_value = function(value) { return _qvr_qiec_is_binding(value) ? value.value : value; };
var _qvr_qiec_constructor = function(constructor, staticArguments, fields, resultType, staticEnvironment) {
  return Object.freeze({ qiec: "constructor", constructor: constructor, static_arguments: _qvr_qiec_specialize(staticArguments, staticEnvironment), fields: Object.freeze(fields), result_type: _qvr_qiec_specialize(resultType, staticEnvironment) });
};
var _qvr_qiec_evidence = function(evidence, staticEnvironment) { return Object.freeze({ qiec: "evidence", evidence: _qvr_qiec_specialize(evidence, staticEnvironment) }); };
var _qvr_qiec_transport = function(evidence, value, targetType, staticEnvironment) { _qvr_qiec_specialize(evidence, staticEnvironment); _qvr_qiec_specialize(targetType, staticEnvironment); return value; };
var _qvr_qiec_attachment = function(attachments, attachment, expectedType, staticEnvironment) {
  if (!Object.prototype.hasOwnProperty.call(attachments, attachment)) { throw new Error("missing QIEC attachment " + attachment); }
  var binding = attachments[attachment];
  if (!_qvr_qiec_is_binding(binding) || !Object.prototype.hasOwnProperty.call(binding, "type")) { throw new Error("QIEC attachment must be a typed binding descriptor"); }
  expectedType = _qvr_qiec_specialize(expectedType, staticEnvironment);
  if (JSON.stringify(binding.type) !== JSON.stringify(expectedType)) { throw new Error("QIEC attachment type disagrees with checked IR"); }
  return _qvr_qiec_value(_qvr_qiec_validate(binding.validator, binding, "attachment " + attachment));
};
var _qvr_qiec_case = function(value, branches, metadata, staticEnvironment) {
  metadata = _qvr_qiec_specialize(metadata, staticEnvironment);
  if (!value || value.qiec !== "constructor") { throw new Error("QIEC case scrutinee is not a constructor value"); }
  if (!Object.prototype.hasOwnProperty.call(metadata.branches, value.constructor)) { throw new Error("constructor is not admitted by the checked QIEC case"); }
  if (!Object.prototype.hasOwnProperty.call(branches, value.constructor)) { throw new Error("no QIEC case branch for " + value.constructor); }
  var branch = metadata.branches[value.constructor];
  var canonical = branch.static_arguments || [];
  var actual = value.static_arguments || [];
  if (actual.length < canonical.length) { throw new Error("QIEC constructor static arguments do not match case branch"); }
  var branchEnvironment = Object.assign({}, staticEnvironment);
  canonical.forEach(function(variable, index) {
    var staticKey = variable.identity || variable.name;
    if (!staticKey) { throw new Error("QIEC case branch has an unbindable static variable"); }
    branchEnvironment[staticKey] = actual[actual.length - canonical.length + index];
  });
  return branches[value.constructor].apply(null, [branchEnvironment].concat(value.fields));
};
var _qvr_qiec_member = function(container, key, fallback) {
  return container && Object.prototype.hasOwnProperty.call(container, key) ? container[key] : fallback;
};
var _qvr_qiec_as_computation = function(value) {
  return value && (value.tag === "pure" || value.tag === "effect" || value.tag === "call" || value.tag === "bind") ? value : _qvr_qiec_pure(value);
};
var _qvr_qiec_validate = function(validator, value, label) {
  if (!validator) { return value; }
  if (validator(_qvr_qiec_value(value)) === false) { throw new Error("QIEC runtime value failed " + label); }
  return value;
};
var _qvr_qiec_validate_computation = function(comp, validator, label) {
  return _qvr_qiec_bind(_qvr_qiec_as_computation(comp), function(value) {
    return _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label));
  });
};
var _qvr_qiec_capture = function(locals, attachmentIds, attachments) {
  var captured = Object.keys(locals).map(function(name) { return { kind: "local", label: name, binding: locals[name] }; });
  attachmentIds.forEach(function(attachment) {
    if (!Object.prototype.hasOwnProperty.call(attachments, attachment)) { throw new Error("missing QIEC attachment " + attachment); }
    captured.push({ kind: "attachment", label: attachment, binding: attachments[attachment] });
  });
  return captured;
};
var _qvr_qiec_duplicable = function(value) {
  if (_qvr_qiec_is_binding(value)) { return value.duplicable === true; }
  if (value === null || ["boolean", "number", "string", "undefined"].indexOf(typeof value) >= 0) { return true; }
  if (Array.isArray(value)) { return Object.isFrozen(value) && value.every(_qvr_qiec_duplicable); }
  if (value && value.qiec === "constructor") { return value.fields.every(_qvr_qiec_duplicable); }
  return value && value.qiec === "evidence";
};
var _qvr_qiec_validate_capture = function(request, currentHandler) {
  (request.captures || []).forEach(function(capture) {
    var binding = capture.binding;
    if (!_qvr_qiec_duplicable(binding)) { throw new Error("QIEC unrestricted resumption captures nonduplicable " + capture.kind + " " + capture.label); }
    if (_qvr_qiec_is_binding(binding)) {
      capture.seed = binding.value;
      if (binding.mutable && typeof binding.fork !== "function") { throw new Error("QIEC mutable duplicable binding requires fork " + capture.label); }
    }
  });
  _qvr_qiec_unique((request.handler_captures || []).concat([currentHandler])).forEach(function(controller) {
    controller.snapshot();
    var handler = controller.get();
    if (!handler.duplicable_context) { throw new Error("QIEC unrestricted resumption captures nonduplicable handler context"); }
    if (handler.mutable_context && typeof handler.fork_context !== "function") { throw new Error("QIEC mutable duplicable handler context requires fork_context"); }
  });
};
var _qvr_qiec_fork_capture = function(request, currentHandler, shot) {
  (request.captures || []).forEach(function(capture) {
    var binding = capture.binding;
    if (_qvr_qiec_is_binding(binding) && typeof binding.fork === "function") { binding.value = binding.fork(capture.seed); }
  });
  _qvr_qiec_unique((request.handler_captures || []).concat([currentHandler])).forEach(function(controller) {
    controller.installFork();
  });
};
var _qvr_qiec_readdress = function(comp, shot) {
  comp = _qvr_qiec_force(comp);
  if (comp.tag === "pure") { return comp; }
  var request = Object.assign({}, comp.request);
  request.address = [request.address[0], request.address[1], request.address[2].concat([shot])];
  return { tag: "effect", request: request, continuation: function(value) { return _qvr_qiec_readdress(comp.continuation(value), shot); } };
};
var _qvr_qiec_validate_handler = function(handler, manifest) {
  if (handler.definition && JSON.stringify(handler.definition) !== JSON.stringify(manifest)) { throw new Error("QIEC runtime handler definition disagrees with checked IR"); }
  var structural = manifest.clauses.map(function(clause) { return clause.operation; }).sort();
  var executable = Object.keys(handler.operations || {}).sort();
  if (JSON.stringify(structural) !== JSON.stringify(executable)) { throw new Error("QIEC runtime handler clauses disagree with checked IR"); }
  if (typeof handler.fork_context === "function" && !handler.duplicable_context) { throw new Error("QIEC handler fork_context requires duplicable_context"); }
  if (handler.mutable_context && handler.duplicable_context && typeof handler.fork_context !== "function") { throw new Error("QIEC mutable duplicable handler requires fork_context"); }
};
var _qvr_qiec_lifecycle = function(handler) { return { handler: handler, finalized: false }; };
var _qvr_qiec_lifecycle_exit = function(lifecycle) {
  if (lifecycle.finalized) { return; }
  lifecycle.finalized = true;
  if (typeof lifecycle.handler.on_exit === "function") { lifecycle.handler.on_exit(); }
};
var _qvr_qiec_lifecycle_drop = function(lifecycle) {
  if (lifecycle.finalized) { return; }
  lifecycle.finalized = true;
  if (typeof lifecycle.handler.on_drop === "function") { lifecycle.handler.on_drop(); }
};
var _qvr_qiec_lifecycle_enter = function(handler) {
  var lifecycle = _qvr_qiec_lifecycle(handler);
  try { if (typeof handler.on_enter === "function") { handler.on_enter(); } }
  catch (error) { _qvr_qiec_lifecycle_drop(lifecycle); throw error; }
  return lifecycle;
};
var _qvr_qiec_drop_request = function(request) {
  (request.lifecycles || []).forEach(_qvr_qiec_lifecycle_drop);
};
var _qvr_qiec_finalize = function(comp, lifecycle) {
  var current = _qvr_qiec_force(_qvr_qiec_as_computation(comp));
  if (current.tag === "pure") { _qvr_qiec_lifecycle_exit(lifecycle); return current; }
  var request = Object.assign({}, current.request);
  request.lifecycles = (request.lifecycles || []).concat([lifecycle]);
  return { tag: "effect", request: request, continuation: function(value) { return _qvr_qiec_finalize(current.continuation(value), lifecycle); } };
};
var _qvr_qiec_invoke = function(entry, args) { return (entry.invoke || entry).apply(null, args); };
var _qvr_qiec_handle = function(comp, instance, manifest, staticArguments, computationStaticEnvironment, handlers, attachments, operations) {
  staticArguments = _qvr_qiec_specialize(staticArguments, computationStaticEnvironment);
  var handlerStaticEnvironment = _qvr_qiec_static_environment(manifest.telescope, staticArguments);
  manifest = _qvr_qiec_specialize(manifest, handlerStaticEnvironment);
  manifest.telescope = [];
  var handlerId = manifest.id;
  var prototype;
  if (Object.prototype.hasOwnProperty.call(handlers, handlerId)) { prototype = handlers[handlerId]; }
  else if (Object.prototype.hasOwnProperty.call(_qvr_qiec_authored, handlerId)) { prototype = _qvr_qiec_authored[handlerId]; }
  else { throw new Error("missing QIEC handler attachment " + handlerId); }
  var handler = prototype;
  if (typeof prototype.context_factory === "function") { handler = prototype.context_factory(); }
  if (handler === prototype && prototype.mutable_context) { throw new Error("QIEC mutable handler requires context_factory"); }
  var handlerRef = { value: handler };
  _qvr_qiec_validate_handler(handlerRef.value, manifest);
  var lifecycleRef = { value: _qvr_qiec_lifecycle_enter(handler) };
  var rootLifecycle = lifecycleRef.value;
  var seedRef = { value: handlerRef.value };
  var controller = {
    get: function() { return handlerRef.value; },
    seed: function() { return seedRef.value; },
    snapshot: function() { seedRef.value = handlerRef.value; },
    capture: function() { return [handlerRef.value, lifecycleRef.value]; },
    restore: function(state) { handlerRef.value = state[0]; lifecycleRef.value = state[1]; },
    installFork: function() {
      var seed = seedRef.value;
      if (typeof seed.fork_context !== "function") { return; }
      var clone = seed.fork_context(seed);
      try { _qvr_qiec_validate_handler(clone, manifest); }
      catch (error) { if (typeof clone.on_drop === "function") { clone.on_drop(); } throw error; }
      handlerRef.value = clone;
      lifecycleRef.value = _qvr_qiec_lifecycle_enter(clone);
    }
  };
  var grades = {};
  manifest.clauses.forEach(function(clause) { grades[clause.operation] = clause.grade; });
  var context = { handler: handlerId, definition: manifest, static_arguments: staticArguments, "static": handlerStaticEnvironment, attachments: attachments || {}, handlers: handlers, operations: operations || {}, resumption_uses: 0 };
  var walk = function(current) {
    var handler = handlerRef.value;
    current = _qvr_qiec_force(current);
    if (current.tag === "pure") {
      var value = _qvr_qiec_validate(handler.input_validator, current.value, "handler input type");
      var result = handler.return ? _qvr_qiec_invoke(handler.return, [value, context]) : value;
      return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, handler.output_validator, "handler output type"), lifecycleRef.value);
    }
    var request = current.request;
    (request.handler_captures || []).forEach(function(capturedController, index) {
      if (capturedController === controller) { controller.restore(request.handler_states[index]); handler = handlerRef.value; }
    });
    var forward = function() {
      var forwardedHandler = handlerRef.value;
      var forwardedLifecycle = lifecycleRef.value;
      var forwarded = Object.assign({}, request);
      forwarded.handler_captures = _qvr_qiec_unique((request.handler_captures || []).concat([controller]));
      forwarded.handler_states = forwarded.handler_captures.map(function(captured) { return captured.capture(); });
      forwarded.lifecycles = (request.lifecycles || []).concat([lifecycleRef.value]);
      return { tag: "effect", request: forwarded, continuation: function(value) {
        handlerRef.value = forwardedHandler;
        lifecycleRef.value = forwardedLifecycle;
        return _qvr_qiec_with_ambient(controller, function() { return walk(current.continuation(value)); });
      } };
    };
    if (request.instance !== instance) { return forward(); }
    var clause = (handler.operations || {})[request.operation];
    if (JSON.stringify(request.effect) !== JSON.stringify(manifest.effect)) { throw new Error("QIEC request and installed handler effect disagree"); }
    if (!clause) {
      if (!manifest.total || manifest.forwards_unknown) { return forward(); }
      throw new Error("missing QIEC handler clause " + request.operation);
    }
    var uses = 0;
    var grade = grades[request.operation];
    if (grade === "omega") { _qvr_qiec_validate_capture(request, controller); }
    var resume = function(value) {
      var shot = uses;
      uses += 1;
      context.resumption_uses = uses;
      if (grade === "0") { throw new Error("QIEC zero-grade clause resumed"); }
      if ((grade === "aff" || grade === "1") && uses > 1) { throw new Error("QIEC resumption exceeds grade " + grade); }
      if (grade === "omega") { _qvr_qiec_fork_capture(request, controller, shot); }
      value = _qvr_qiec_validate(clause.result_validator, value, "handler operation result type");
      var resumed = _qvr_qiec_with_ambient(controller, function() { return walk(_qvr_qiec_readdress(current.continuation(value), shot)); });
      return resumed.tag === "pure" ? resumed.value : resumed;
    };
    var clauseLifecycle = lifecycleRef.value;
    var result;
    try { result = _qvr_qiec_invoke(clause, [request, resume, context]); }
    catch (error) { _qvr_qiec_drop_request(request); _qvr_qiec_lifecycle_drop(clauseLifecycle); throw error; }
    if (grade === "1" && uses !== 1) { _qvr_qiec_drop_request(request); _qvr_qiec_lifecycle_drop(clauseLifecycle); throw new Error("QIEC linear clause must resume exactly once"); }
    if (uses === 0) { _qvr_qiec_drop_request(request); }
    return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, handler.output_validator, "handler output type"), clauseLifecycle);
  };
  try { comp = _qvr_qiec_with_ambient(controller, function() { return _qvr_qiec_force(comp()); }); return walk(comp); }
  catch (error) { _qvr_qiec_lifecycle_drop(lifecycleRef.value); _qvr_qiec_lifecycle_drop(rootLifecycle); throw error; }
};
var _qvr_qiec_run = function(build, operations) {
  _qvr_qiec_serials.call = 0;
  _qvr_qiec_serials.instance = 0;
  var current = _qvr_qiec_force(build());
  while (current.tag === "effect") {
    var request = current.request;
    var entry = operations[request.instance + "|" + request.operation];
    if (!entry && operations[request.instance]) { entry = operations[request.instance][request.operation]; }
    if (!entry) { throw new Error("unhandled QIEC operation " + request.operation + " on " + request.instance); }
    try {
      var result = _qvr_qiec_invoke(entry, [request]);
      current = _qvr_qiec_force(current.continuation(_qvr_qiec_validate(entry.result_validator, result, "operation result type")));
    } catch (error) { _qvr_qiec_drop_request(request); throw error; }
  }
  return _qvr_qiec_value(current.value);
};
