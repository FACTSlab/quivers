// WebPPL runtime helpers: distribution constructors for QVR families
// that WebPPL's built-in `dists` module does not ship.
//
// Each helper is a plain JavaScript function that takes a `params`
// object and returns a distribution object with `sample`, `score`,
// and `support` methods. WebPPL's `sample(dist)` and `observe(dist,
// value)` accept any object exposing this triple.
//
// Mathematical references for each density:
//
//   Logistic              : Balakrishnan 1991
//                           https://doi.org/10.1201/9781482277098
//   BetaBinomial          : Skellam 1948
//                           https://www.jstor.org/stable/2983694
//   HalfStudentT          : Gelman 2006
//                           https://doi.org/10.1214/06-BA117A
//   Kumaraswamy           : Kumaraswamy 1980
//                           https://doi.org/10.1016/0022-1694(80)90036-0
//   LKJCholesky           : Lewandowski, Kurowicka, Joe 2009
//                           https://doi.org/10.1016/j.jmva.2009.04.008
//   ContinuousBernoulli   : Loaiza-Ganem and Cunningham 2019
//                           https://arxiv.org/abs/1907.06845

var _lgamma = function(z) {
  // Lanczos approximation. Domain: z > 0.
  var g = 7;
  var c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - _lgamma(1 - z);
  }
  z = z - 1;
  var x = c[0];
  var i = 1;
  while (i < g + 2) {
    x = x + c[i] / (z + i);
    i = i + 1;
  }
  var t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
};

var _lbeta = function(a, b) {
  return _lgamma(a) + _lgamma(b) - _lgamma(a + b);
};

var _gaussian_sample = function(mu, sigma) {
  // Box-Muller.
  var u1 = Math.random();
  var u2 = Math.random();
  var z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
};

var _gamma_sample = function(shape, scale) {
  // Marsaglia and Tsang 2000 for shape >= 1; for shape < 1 boost.
  if (shape < 1) {
    var u = Math.random();
    return _gamma_sample(shape + 1, scale) * Math.pow(u, 1 / shape);
  }
  var d = shape - 1 / 3;
  var c = 1 / Math.sqrt(9 * d);
  while (true) {
    var x = _gaussian_sample(0, 1);
    var v = 1 + c * x;
    if (v <= 0) { continue; }
    v = v * v * v;
    var u2 = Math.random();
    if (u2 < 1 - 0.0331 * x * x * x * x) {
      return d * v * scale;
    }
    if (Math.log(u2) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
};

var _beta_sample = function(a, b) {
  var x = _gamma_sample(a, 1);
  var y = _gamma_sample(b, 1);
  return x / (x + y);
};

var _binomial_sample = function(n, p) {
  var k = 0;
  var i = 0;
  while (i < n) {
    if (Math.random() < p) { k = k + 1; }
    i = i + 1;
  }
  return k;
};

var Logistic = function(params) {
  var loc = params.loc;
  var scale = params.scale;
  return {
    sample: function() {
      var u = Math.random();
      return loc + scale * Math.log(u / (1 - u));
    },
    score: function(x) {
      var z = (x - loc) / scale;
      return -z - Math.log(scale) - 2 * Math.log(1 + Math.exp(-z));
    },
    support: function() {
      return { lower: -Infinity, upper: Infinity };
    }
  };
};

var BetaBinomial = function(params) {
  var n = params.total_count;
  var a = params.concentration1;
  var b = params.concentration0;
  return {
    sample: function() {
      var p = _beta_sample(a, b);
      return _binomial_sample(n, p);
    },
    score: function(k) {
      var log_comb = _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1);
      var log_beta_post = _lbeta(a + k, b + n - k);
      var log_beta_prior = _lbeta(a, b);
      return log_comb + log_beta_post - log_beta_prior;
    },
    support: function() {
      return { lower: 0, upper: n };
    }
  };
};

var HalfStudentT = function(params) {
  var df = params.df;
  var scale = params.scale;
  return {
    sample: function() {
      var z = _gaussian_sample(0, 1);
      var u = _gamma_sample(df / 2, 2);
      var t = z / Math.sqrt(u / df);
      return Math.abs(t) * scale;
    },
    score: function(x) {
      if (x < 0) { return -Infinity; }
      var z = x / scale;
      var log_t = _lgamma((df + 1) / 2) - _lgamma(df / 2)
                  - 0.5 * Math.log(df * Math.PI)
                  - ((df + 1) / 2) * Math.log(1 + (z * z) / df);
      return log_t - Math.log(scale) + Math.log(2);
    },
    support: function() {
      return { lower: 0, upper: Infinity };
    }
  };
};

var Kumaraswamy = function(params) {
  var a = params.concentration1;
  var b = params.concentration0;
  return {
    sample: function() {
      var u = Math.random();
      return Math.pow(1 - Math.pow(1 - u, 1 / b), 1 / a);
    },
    score: function(x) {
      if (x <= 0 || x >= 1) { return -Infinity; }
      return Math.log(a) + Math.log(b)
             + (a - 1) * Math.log(x)
             + (b - 1) * Math.log(1 - Math.pow(x, a));
    },
    support: function() {
      return { lower: 0, upper: 1 };
    }
  };
};

var LKJCholesky = function(params) {
  var dim = params.dim;
  var eta = params.concentration;
  return {
    sample: function() {
      // Onion method: draw beta-distributed partial correlations
      // and stitch them into a Cholesky factor. Returns a square
      // lower-triangular matrix as an array of arrays.
      var L = [];
      var i = 0;
      while (i < dim) {
        var row = [];
        var j = 0;
        while (j < dim) { row.push(0); j = j + 1; }
        L.push(row);
        i = i + 1;
      }
      L[0][0] = 1;
      var k = 1;
      while (k < dim) {
        var alpha = eta + (dim - 1 - k) / 2;
        var r2 = _beta_sample(k / 2, alpha);
        var r = Math.sqrt(r2);
        // Sample a unit vector in R^k via independent normals.
        var u = [];
        var sumsq = 0;
        var m = 0;
        while (m < k) {
          var g = _gaussian_sample(0, 1);
          u.push(g);
          sumsq = sumsq + g * g;
          m = m + 1;
        }
        var norm = Math.sqrt(sumsq);
        var t = 0;
        while (t < k) {
          L[k][t] = r * u[t] / norm;
          t = t + 1;
        }
        L[k][k] = Math.sqrt(1 - r2);
        k = k + 1;
      }
      return L;
    },
    score: function(L) {
      // Unnormalised LKJ density on the Cholesky factor:
      //   log p(L) = sum_{i=2}^{dim} (dim - i + 2 * eta - 2) * log L[i][i]
      // plus the LKJ normalising constant. For the syntactic check
      // we return only the data-dependent part; the constant cancels
      // in MCMC ratios.
      var lp = 0;
      var i = 1;
      while (i < dim) {
        lp = lp + (dim - i - 1 + 2 * eta - 2) * Math.log(L[i][i]);
        i = i + 1;
      }
      return lp;
    },
    support: function() {
      return { lower: -1, upper: 1 };
    }
  };
};

var ContinuousBernoulli = function(params) {
  var p = params.probs;
  var log_norm = (function() {
    if (Math.abs(p - 0.5) < 1e-4) {
      var d = p - 0.5;
      return Math.log(2) + 2 * d * d + (4 / 3) * d * d * d * d;
    }
    return Math.log(Math.abs(2 * Math.atanh(1 - 2 * p)))
           - Math.log(Math.abs(1 - 2 * p));
  })();
  return {
    sample: function() {
      var u = Math.random();
      if (Math.abs(p - 0.5) < 1e-4) { return u; }
      return Math.log1p((2 * p - 1) * u / (1 - p)) / Math.log(p / (1 - p));
    },
    score: function(x) {
      if (x < 0 || x > 1) { return -Infinity; }
      return x * Math.log(p) + (1 - x) * Math.log(1 - p) + log_norm;
    },
    support: function() {
      return { lower: 0, upper: 1 };
    }
  };
};

var _qvr_rbf_kernel = function(x, length_scale, jitter) {
  var n = x.length;
  var K = [];
  for (var i = 0; i < n; i++) {
    var row = [];
    for (var j = 0; j < n; j++) {
      var d = x[i] - x[j];
      var v = Math.exp(-0.5 * d * d / (length_scale * length_scale));
      if (i === j) { v += jitter; }
      row.push(v);
    }
    K.push(row);
  }
  return K;
};

var _qvr_zeros = function(n) {
  var v = [];
  for (var i = 0; i < n; i++) { v.push(0); }
  return v;
};
