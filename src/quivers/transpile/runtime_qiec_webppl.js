// QIEC distribution bridge for generated WebPPL programs. A QIEC tensor is
// a frozen nested array until it reaches a distribution, where it becomes
// a plain nested array; a log density is the distribution's own `score`
// at the converted value. Half-line families fold a symmetric base onto
// the nonnegative reals, doubling its density there.
var _qvr_qiec_array = function(value) {
  value = _qvr_qiec_value(value);
  if (Array.isArray(value)) { return value.map(_qvr_qiec_array); }
  if (typeof value === "boolean") { return value ? 1 : 0; }
  return value;
};
var _qvr_qiec_log_density = function(distribution, value) {
  return distribution.score(_qvr_qiec_array(value));
};
var _qvr_qiec_half = function(base) {
  return {
    sample: function() { return Math.abs(base.sample()); },
    score: function(x) { return x < 0 ? -Infinity : Math.log(2) + base.score(x); },
    support: function() { return { lower: 0, upper: Infinity }; }
  };
};
var _qvr_qiec_categorical = function(ps) {
  return Categorical({ ps: ps, vs: _qvr_support(ps) });
};
var _qvr_qiec_mixture_normal = function(weights, loc, scale) {
  var components = loc.map(function(m, i) { return Gaussian({ mu: m, sigma: scale[i] }); });
  return Mixture({ dists: components, ps: weights });
};
