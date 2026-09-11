;; Church runtime: a self-contained reference implementation of the
;; Church stochastic-lambda-calculus primitive set and distribution
;; library in plain Scheme.
;;
;; Church has no single canonical interpreter; its target is the
;; abstract semantics of Goodman, Mansinghka, Roy, Bonawitz, and
;; Tenenbaum (2008). This file makes the abstract semantics concrete
;; and self-contained: it defines `sample`, `observe`, and `factor`
;; on top of a distribution-object protocol, plus one correctly
;; parameterised distribution constructor per QVR family that the
;; emitter references. The transpile-time graft parses this file once
;; at module load and copies the resulting subtree into every
;; per-render schema above the model `(define (model ...) ...)` form,
;; so every emitted program is a complete Scheme module.
;;
;; A distribution is a tagged triple `(dist <sampler> <scorer>)` where
;; the sampler is a nullary thunk returning a draw and the scorer maps
;; a value to its log-density. `sample` draws and accumulates the
;; draw's log-density into the trace weight; `observe` accumulates the
;; scored value's log-density; `factor` adds a raw log-weight. The
;; joint log-density of a run is the accumulated `*log-weight*`, which
;; equals the QVR model's joint up to an additive constant.

;; ---- deterministic pseudo-random source -------------------------------
;; A linear congruential generator with mutable state keeps the runtime
;; free of host-specific `random` arity conventions.
(define *rng-state* 123456789)

(define (uniform-random)
  (set! *rng-state* (modulo (+ (* *rng-state* 1103515245) 12345) 2147483648))
  (/ *rng-state* 2147483648.0))

;; ---- broadcasting arithmetic ------------------------------------------
;; The deterministic-let emitter renders QVR arithmetic as plain `(+ a
;; b)`, `(- a b)`, `(* a b)`, `(/ a b)` forms, but those operands may
;; be vectors (a per-axis latent times a per-axis scale, a scalar
;; intercept plus a per-row linear predictor, ...). Shadow the four
;; operators with shape-polymorphic, variadic versions: a binary step
;; over two equal-length lists is element-wise, a scalar against a list
;; broadcasts, and two scalars reduce to the primitive. Each operator
;; stays fully variadic (and reduces left-to-right) so the runtime's
;; own primitive draws -- which call `*` and `+` at three or more
;; arguments -- and `(apply + ...)` folds keep working. The primitives
;; are captured first so the scalar base cases stay exact.
(define %add +)
(define %sub -)
(define %mul *)
(define %div /)

;; Element-wise combination of two operands under a scalar binary op,
;; broadcasting a scalar against a list and recursing into nested
;; lists so matrices combine cell-by-cell.
(define (broadcast2 op a b)
  (cond ((and (pair? a) (pair? b))
         (map (lambda (x y) (broadcast2 op x y)) a b))
        ((pair? a) (map (lambda (x) (broadcast2 op x b)) a))
        ((pair? b) (map (lambda (y) (broadcast2 op a y)) b))
        (else (op a b))))

;; Element-wise unary map over a (possibly nested) operand.
(define (broadcast1 op a)
  (if (pair? a) (map (lambda (x) (broadcast1 op x)) a) (op a)))

(define (+ . args)
  (cond ((null? args) 0)
        ((null? (cdr args)) (car args))
        (else (fold-left (lambda (acc x) (broadcast2 %add acc x))
                         (car args) (cdr args)))))

(define (* . args)
  (cond ((null? args) 1)
        ((null? (cdr args)) (car args))
        (else (fold-left (lambda (acc x) (broadcast2 %mul acc x))
                         (car args) (cdr args)))))

;; Unary `-` negates (element-wise); n-ary `-` subtracts left-to-right.
(define (- . args)
  (cond ((null? args) (error '- "needs at least one argument"))
        ((null? (cdr args)) (broadcast1 (lambda (x) (%sub x)) (car args)))
        (else (fold-left (lambda (acc x) (broadcast2 %sub acc x))
                         (car args) (cdr args)))))

;; Unary `/` reciprocates (element-wise); n-ary `/` divides left-to-right.
(define (/ . args)
  (cond ((null? args) (error '/ "needs at least one argument"))
        ((null? (cdr args)) (broadcast1 (lambda (x) (%div x)) (car args)))
        (else (fold-left (lambda (acc x) (broadcast2 %div acc x))
                         (car args) (cdr args)))))

;; ---- numeric constants and helpers ------------------------------------
(define *pi* 3.141592653589793)
(define *neg-inf* (/ -1.0 0.0))

(define (square x) (* x x))

;; Length-n list of the integers 0 .. n-1.
(define (iota n)
  (let loop ((i (- n 1)) (acc (quote ())))
    (if (< i 0) acc (loop (- i 1) (cons i acc)))))

;; Length-n list whose every element is v.
(define (make-list n v)
  (map (lambda (i) v) (iota n)))

;; Length-n list produced by calling the thunk n times.
(define (repeat n thunk)
  (map (lambda (i) (thunk)) (iota n)))

;; Sum of a list of numbers.
(define (sum lst) (apply + lst))

;; Dot product of two equal-length lists.
(define (dot a b) (apply + (map * a b)))

;; Element-wise list arithmetic.
(define (add-lists a b) (map + a b))

(define (sub-lists a b) (map - a b))

;; First k elements of lst.
(define (take-list lst k)
  (if (= k 0) (quote ()) (cons (car lst) (take-list (cdr lst) (- k 1)))))

;; lst with its first k elements removed.
(define (drop-list lst k)
  (if (= k 0) lst (drop-list (cdr lst) (- k 1))))

;; Row-major flatten of a matrix (list of row lists).
(define (vec mat) (apply append mat))

;; Inverse of vec on a p x n row-major matrix.
(define (reshape lst p n)
  (if (= p 0)
      (quote ())
      (cons (take-list lst n) (reshape (drop-list lst n) (- p 1) n))))

;; Logistic sigmoid.
(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

;; ---- log-gamma via the Lanczos approximation --------------------------
(define *lanczos-c*
  (list 0.99999999999980993
        676.5203681218851
        -1259.1392167224028
        771.32342877765313
        -176.61502916214059
        12.507343278686905
        -0.13857109526572012
        9.9843695780195716e-6
        1.5056327351493116e-7))

(define (log-gamma x)
  (if (< x 0.5)
      ;; Reflection: log G(x) = log(pi / sin(pi x)) - log G(1 - x).
      (- (log (/ *pi* (sin (* *pi* x)))) (log-gamma (- 1.0 x)))
      (let ((x1 (- x 1.0)))
        (let ((tt (+ x1 7.5)))
          (let acc-loop ((i 1) (rest (cdr *lanczos-c*)) (a (car *lanczos-c*)))
            (if (null? rest)
                (+ (* 0.5 (log (* 2.0 *pi*)))
                   (* (+ x1 0.5) (log tt))
                   (- tt)
                   (log a))
                (acc-loop (+ i 1)
                          (cdr rest)
                          (+ a (/ (car rest) (+ x1 i))))))))))

(define (log-factorial n) (log-gamma (+ n 1.0)))

;; ---- primitive draws --------------------------------------------------
;; Standard normal via the Box-Muller transform.
(define (draw-standard-normal)
  (* (sqrt (* -2.0 (log (uniform-random))))
     (cos (* 2.0 *pi* (uniform-random)))))

(define (draw-normal mu sigma) (+ mu (* sigma (draw-standard-normal))))

;; Gamma(shape, rate=1) via Marsaglia and Tsang (2000).
(define (draw-gamma-unit shape)
  (if (< shape 1.0)
      (* (draw-gamma-unit (+ shape 1.0))
         (expt (uniform-random) (/ 1.0 shape)))
      (let ((d (- shape (/ 1.0 3.0))))
        (let ((c (/ 1.0 (sqrt (* 9.0 d)))))
          (let reject ()
            (let ((z (draw-standard-normal)))
              (let ((v (* (+ 1.0 (* c z)) (+ 1.0 (* c z)) (+ 1.0 (* c z)))))
                (if (<= v 0.0)
                    (reject)
                    (let ((u (uniform-random)))
                      (if (< (log u) (+ (* 0.5 z z) (* d (+ 1.0 (- v) (log v)))))
                          (* d v)
                          (reject)))))))))))

;; Gamma(shape, rate).
(define (draw-gamma shape rate) (/ (draw-gamma-unit shape) rate))

;; Poisson(rate) via Knuth's multiplication method.
(define (draw-poisson rate)
  (let ((threshold (exp (- rate))))
    (let loop ((k 0) (p 1.0))
      (let ((p2 (* p (uniform-random))))
        (if (<= p2 threshold) k (loop (+ k 1) p2))))))

;; Geometric(probs): number of failures before the first success.
(define (draw-geometric p)
  (floor (/ (log (uniform-random)) (log (- 1.0 p)))))

;; Categorical index draw from a list of probabilities.
(define (draw-categorical probs)
  (let ((u (uniform-random)))
    (let loop ((i 0) (ps probs) (acc 0.0))
      (let ((acc2 (+ acc (car ps))))
        (if (or (< u acc2) (null? (cdr ps)))
            i
            (loop (+ i 1) (cdr ps) acc2))))))

;; ---- linear algebra for the multivariate normal -----------------------
;; Element (i, j) of a matrix stored as a list of row lists.
(define (mref m i j) (list-ref (list-ref m i) j))

;; The main diagonal of an n x n matrix.
(define (diagonal m)
  (map (lambda (i) (mref m i i)) (iota (length m))))

;; Matrix (list of rows) times vector.
(define (mat-vec m v) (map (lambda (row) (dot row v)) m))

;; Kronecker product A (x) B of two matrices stored as lists of rows.
;; Result row (i, k) at column (j, l) is A[i][j] * B[k][l].
(define (mat-kron a b)
  (apply append
         (map (lambda (a-row)
                (map (lambda (b-row)
                       (apply append
                              (map (lambda (a-ij)
                                     (map (lambda (x) (* x a-ij)) b-row))
                                   a-row)))
                     b))
              a)))

;; Lower-triangular Cholesky factor L of an SPD matrix a, with
;; a = L L^T. Each returned row is padded with zeros to length n.
(define (cholesky a)
  (let ((n (length a)))
    (let build ((i 0) (l (quote ())))
      (if (= i n)
          l
          (build (+ i 1) (append l (list (cholesky-row a l i n))))))))

(define (cholesky-row a l i n)
  (let jloop ((j 0) (row (quote ())))
    (cond
      ((> i j)
       (let ((lj (list-ref l j)))
         (let ((val (/ (- (mref a i j) (dot row (take-list lj j)))
                       (list-ref lj j))))
           (jloop (+ j 1) (append row (list val))))))
      ((= i j)
       (append row (list (sqrt (- (mref a i i) (sum (map square row)))))))
      (else
       (append row (make-list (- n j) 0.0))))))

;; Solve L y = b for y with L lower-triangular.
(define (forward-solve l b)
  (let ((n (length b)))
    (let solve ((i 0) (y (quote ())))
      (if (= i n)
          y
          (let ((li (list-ref l i)))
            (let ((yi (/ (- (list-ref b i) (dot y (take-list li (length y))))
                         (list-ref li i))))
              (solve (+ i 1) (append y (list yi)))))))))

;; ---- distribution-object protocol and trace primitives ----------------
(define (make-dist sampler scorer) (list (quote dist) sampler scorer))

(define (dist-draw d) ((cadr d)))

(define (dist-score d x) ((caddr d) x))

(define *log-weight* 0.0)

(define (record-score! s) (set! *log-weight* (+ *log-weight* s)))

;; Draw a latent value and accumulate its log-density.
(define (sample d)
  (let ((x (dist-draw d)))
    (record-score! (dist-score d x))
    x))

;; Score an observed value against a distribution.
(define (observe d x)
  (record-score! (dist-score d x))
  x)

;; Add a raw log-weight increment.
(define (factor s)
  (record-score! s)
  s)

;; Restrict a distribution that is symmetric about zero to the
;; nonnegative reals, folding its mass by absolute value. The folded
;; density on x >= 0 is twice the base density, so `(half (gaussian 0
;; s))` is exactly HalfNormal(s) and `(half (cauchy 0 s))` is exactly
;; HalfCauchy(s).
(define (half base)
  (make-dist
    (lambda () (abs (dist-draw base)))
    (lambda (x)
      (if (< x 0.0)
          *neg-inf*
          (+ (log 2.0) (dist-score base x))))))

;; ---- scalar distributions ---------------------------------------------
(define (gaussian mu sigma)
  (make-dist
    (lambda () (draw-normal mu sigma))
    (lambda (x)
      (- (* -0.5 (square (/ (- x mu) sigma)))
         (log sigma)
         (* 0.5 (log (* 2.0 *pi*)))))))

(define (cauchy loc scale)
  (make-dist
    (lambda () (+ loc (* scale (tan (* *pi* (- (uniform-random) 0.5))))))
    (lambda (x)
      (- (- (log *pi*))
         (log scale)
         (log (+ 1.0 (square (/ (- x loc) scale))))))))

(define (exponential rate)
  (make-dist
    (lambda () (/ (- (log (uniform-random))) rate))
    (lambda (x) (- (log rate) (* rate x)))))

(define (gamma shape rate)
  (make-dist
    (lambda () (draw-gamma shape rate))
    (lambda (x)
      (+ (* shape (log rate))
         (* (- shape 1.0) (log x))
         (- (* rate x))
         (- (log-gamma shape))))))

(define (beta a b)
  (make-dist
    (lambda ()
      (let ((ga (draw-gamma-unit a)) (gb (draw-gamma-unit b)))
        (/ ga (+ ga gb))))
    (lambda (x)
      (+ (- (log-gamma (+ a b)) (log-gamma a) (log-gamma b))
         (* (- a 1.0) (log x))
         (* (- b 1.0) (log (- 1.0 x)))))))

(define (uniform lo hi)
  (make-dist
    (lambda () (+ lo (* (- hi lo) (uniform-random))))
    (lambda (x) (- (log (- hi lo))))))

(define (lognormal mu sigma)
  (make-dist
    (lambda () (exp (draw-normal mu sigma)))
    (lambda (x)
      (- (* -0.5 (square (/ (- (log x) mu) sigma)))
         (log sigma)
         (* 0.5 (log (* 2.0 *pi*)))
         (log x)))))

;; StudentT(df, loc, scale): torch's location-scale parameterisation.
(define (student-t df loc scale)
  (make-dist
    (lambda ()
      (let ((z (draw-standard-normal))
            (g (draw-gamma (/ df 2.0) 0.5)))
        (+ loc (* scale (/ z (sqrt (/ g df)))))))
    (lambda (x)
      (let ((z (/ (- x loc) scale)))
        (+ (- (log-gamma (/ (+ df 1.0) 2.0)) (log-gamma (/ df 2.0)))
           (* -0.5 (log (* df *pi*)))
           (- (log scale))
           (* (- (/ (+ df 1.0) 2.0)) (log (+ 1.0 (/ (* z z) df)))))))))

;; Weibull(scale, concentration): torch's scale / concentration form.
(define (weibull scale k)
  (make-dist
    (lambda () (* scale (expt (- (log (uniform-random))) (/ 1.0 k))))
    (lambda (x)
      (+ (log k)
         (- (log scale))
         (* (- k 1.0) (- (log x) (log scale)))
         (- (expt (/ x scale) k))))))

;; Pareto(scale, alpha): support x >= scale.
(define (pareto scale alpha)
  (make-dist
    (lambda () (/ scale (expt (uniform-random) (/ 1.0 alpha))))
    (lambda (x)
      (+ (log alpha) (* alpha (log scale)) (* (- (+ alpha 1.0)) (log x))))))

(define (poisson rate)
  (make-dist
    (lambda () (draw-poisson rate))
    (lambda (x) (- (* x (log rate)) rate (log-factorial x)))))

;; Geometric(probs): torch's failures-before-success support.
(define (geometric p)
  (make-dist
    (lambda () (draw-geometric p))
    (lambda (x) (+ (* x (log (- 1.0 p))) (log p)))))

;; NegativeBinomial(total_count, probs): torch's Gamma-Poisson mixture,
;; pmf(k) = C(k + r - 1, k) (1 - probs)^r probs^k.
(define (negative-binomial r p)
  (make-dist
    (lambda () (draw-poisson (draw-gamma r (/ (- 1.0 p) p))))
    (lambda (x)
      (+ (- (log-gamma (+ x r)) (log-gamma r) (log-factorial x))
         (* r (log (- 1.0 p)))
         (* x (log p))))))

;; Bernoulli as a numeric 0/1 draw so observed integer data scores
;; against the same coding rather than a boolean flip.
(define (flip p)
  (make-dist
    (lambda () (if (< (uniform-random) p) 1 0))
    (lambda (x) (if (> x 0.5) (log p) (log (- 1.0 p))))))

(define (categorical probs)
  (make-dist
    (lambda () (draw-categorical probs))
    (lambda (x) (log (list-ref probs x)))))

;; ---- vector distributions ---------------------------------------------
(define (dirichlet alphas)
  (make-dist
    (lambda ()
      (let ((gs (map draw-gamma-unit alphas)))
        (let ((s (sum gs)))
          (map (lambda (g) (/ g s)) gs))))
    (lambda (x)
      (+ (log-gamma (sum alphas))
         (- (sum (map log-gamma alphas)))
         (sum (map (lambda (a xi) (* (- a 1.0) (log xi))) alphas x))))))

(define (multivariate-gaussian mean cov)
  (make-dist
    (lambda ()
      (let ((l (cholesky cov))
            (z (map (lambda (i) (draw-standard-normal)) mean)))
        (add-lists mean (mat-vec l z))))
    (lambda (x)
      (let ((l (cholesky cov))
            (n (length mean)))
        (let ((y (forward-solve l (sub-lists x mean))))
          (* -0.5
             (+ (* n (log (* 2.0 *pi*)))
                (* 2.0 (sum (map log (diagonal l))))
                (dot y y))))))))

;; ---- matrix distribution ----------------------------------------------
;; MatrixNormal(mu, U, V): X in R^{p x n} with row covariance U (p x p)
;; and column covariance V (n x n). The row-major flatten vec(X) is
;; MultiNormal(vec(mu), U (x) V), so the emitted object samples the
;; flattened vector and reshapes it back to a p x n matrix, and scores
;; an observed matrix through the same identity.
(define (matrix-normal mu u v)
  (let ((p (length mu))
        (n (length (car mu))))
    (let ((base (multivariate-gaussian (vec mu) (mat-kron u v))))
      (make-dist
        (lambda () (reshape (dist-draw base) p n))
        (lambda (x) (dist-score base (vec x)))))))
