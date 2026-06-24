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
//   MatrixNormal          : Dawid 1981
//                           https://doi.org/10.1093/biomet/68.1.265

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

// Dense matrix utilities. Matrices are arrays-of-arrays in row-major
// order (so `A[i][j]` is row i, column j). The helpers cover only what
// MatrixNormal needs: element-wise add / sub, transpose, multiply,
// Cholesky decomposition (lower-triangular), Cholesky-based solve, log
// determinant, and trace.
var _mat_add = function(A, B) {
  var p = A.length;
  var n = A[0].length;
  var C = [];
  var i = 0;
  while (i < p) {
    var row = [];
    var j = 0;
    while (j < n) { row.push(A[i][j] + B[i][j]); j = j + 1; }
    C.push(row);
    i = i + 1;
  }
  return C;
};

var _mat_sub = function(A, B) {
  var p = A.length;
  var n = A[0].length;
  var C = [];
  var i = 0;
  while (i < p) {
    var row = [];
    var j = 0;
    while (j < n) { row.push(A[i][j] - B[i][j]); j = j + 1; }
    C.push(row);
    i = i + 1;
  }
  return C;
};

var _mat_transpose = function(A) {
  var p = A.length;
  var n = A[0].length;
  var T = [];
  var i = 0;
  while (i < n) {
    var row = [];
    var j = 0;
    while (j < p) { row.push(A[j][i]); j = j + 1; }
    T.push(row);
    i = i + 1;
  }
  return T;
};

var _mat_mul = function(A, B) {
  var p = A.length;
  var k = A[0].length;
  var n = B[0].length;
  var C = [];
  var i = 0;
  while (i < p) {
    var row = [];
    var j = 0;
    while (j < n) {
      var s = 0;
      var t = 0;
      while (t < k) { s = s + A[i][t] * B[t][j]; t = t + 1; }
      row.push(s);
      j = j + 1;
    }
    C.push(row);
    i = i + 1;
  }
  return C;
};

var _mat_chol = function(A) {
  // Standard Cholesky decomposition for SPD A; returns lower
  // triangular L with L L' = A.
  var n = A.length;
  var L = [];
  var ii = 0;
  while (ii < n) {
    var zero_row = [];
    var jz = 0;
    while (jz < n) { zero_row.push(0); jz = jz + 1; }
    L.push(zero_row);
    ii = ii + 1;
  }
  var i = 0;
  while (i < n) {
    var j = 0;
    while (j <= i) {
      var s = 0;
      var k = 0;
      while (k < j) { s = s + L[i][k] * L[j][k]; k = k + 1; }
      if (i == j) {
        L[i][j] = Math.sqrt(A[i][i] - s);
      } else {
        L[i][j] = (A[i][j] - s) / L[j][j];
      }
      j = j + 1;
    }
    i = i + 1;
  }
  return L;
};

var _mat_solve_chol = function(L, B) {
  // Solve A X = B given the Cholesky factor L of A (A = L L').
  // Forward substitution L Y = B, then backward L' X = Y.
  var n = L.length;
  var m = B[0].length;
  var Y = [];
  var yi = 0;
  while (yi < n) {
    var yrow = [];
    var yj = 0;
    while (yj < m) { yrow.push(0); yj = yj + 1; }
    Y.push(yrow);
    yi = yi + 1;
  }
  var i = 0;
  while (i < n) {
    var col = 0;
    while (col < m) {
      var s = B[i][col];
      var k = 0;
      while (k < i) { s = s - L[i][k] * Y[k][col]; k = k + 1; }
      Y[i][col] = s / L[i][i];
      col = col + 1;
    }
    i = i + 1;
  }
  var X = [];
  var xi = 0;
  while (xi < n) {
    var xrow = [];
    var xj = 0;
    while (xj < m) { xrow.push(0); xj = xj + 1; }
    X.push(xrow);
    xi = xi + 1;
  }
  var ib = n - 1;
  while (ib >= 0) {
    var c2 = 0;
    while (c2 < m) {
      var s2 = Y[ib][c2];
      var k2 = ib + 1;
      while (k2 < n) { s2 = s2 - L[k2][ib] * X[k2][c2]; k2 = k2 + 1; }
      X[ib][c2] = s2 / L[ib][ib];
      c2 = c2 + 1;
    }
    ib = ib - 1;
  }
  return X;
};

var _mat_logdet_chol = function(L) {
  // log|A| = 2 sum_i log L[i][i] for the Cholesky factor of A.
  var n = L.length;
  var s = 0;
  var i = 0;
  while (i < n) { s = s + Math.log(L[i][i]); i = i + 1; }
  return 2 * s;
};

var _mat_trace = function(A) {
  var n = A.length;
  var s = 0;
  var i = 0;
  while (i < n) { s = s + A[i][i]; i = i + 1; }
  return s;
};

var MatrixNormal = function(params) {
  // X in R^{p x n} with mean loc, row covariance U (p x p, SPD), and
  // column covariance V (n x n, SPD). Equivalent to
  //   vec(X) ~ MultiNormal(vec(loc), V (x) U).
  // log p(X | loc, U, V) =
  //   -0.5 * ( n p log(2 pi) + n log|U| + p log|V|
  //          + tr(V^{-1} (X - loc)' U^{-1} (X - loc)) )
  var loc = params.loc;
  var U = params.row_covariance;
  var V = params.col_covariance;
  return {
    sample: function() {
      // X = loc + L_U Z L_V' with Z iid standard normal, L_U L_U' = U,
      // L_V L_V' = V.
      var p = loc.length;
      var n = loc[0].length;
      var L_U = _mat_chol(U);
      var L_V = _mat_chol(V);
      var Z = [];
      var i = 0;
      while (i < p) {
        var row = [];
        var j = 0;
        while (j < n) { row.push(_gaussian_sample(0, 1)); j = j + 1; }
        Z.push(row);
        i = i + 1;
      }
      return _mat_add(loc, _mat_mul(_mat_mul(L_U, Z), _mat_transpose(L_V)));
    },
    score: function(X) {
      var p = X.length;
      var n = X[0].length;
      var D = _mat_sub(X, loc);
      var L_U = _mat_chol(U);
      var L_V = _mat_chol(V);
      var U_inv_D = _mat_solve_chol(L_U, D);
      var quad = _mat_mul(_mat_transpose(D), U_inv_D);
      var V_inv_quad = _mat_solve_chol(L_V, quad);
      var trace_term = _mat_trace(V_inv_quad);
      var log_det_U = _mat_logdet_chol(L_U);
      var log_det_V = _mat_logdet_chol(L_V);
      return -0.5 * (n * p * Math.log(2 * Math.PI)
                     + n * log_det_U
                     + p * log_det_V
                     + trace_term);
    },
    support: function() {
      return { lower: -Infinity, upper: Infinity };
    }
  };
};
