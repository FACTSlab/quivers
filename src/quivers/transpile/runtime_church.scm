;; Church runtime helpers: distribution constructors for QVR
;; families that Church's built-in primitive set does not ship.
;;
;; Each helper is a top-level (define ...) form. The transpile-time
;; graft parses this file once at module load and copies the
;; resulting subtree into the per-render schema above the model
;; (define ...) form.

;; Matrix-normal density:
;;   X in R^{p x n} with mean M (p x n), row covariance U (p x p,
;;   SPD), column covariance V (n x n, SPD). Equivalent to
;;     vec(X) ~ MultiNormal(vec(M), V (x) U)
;; where (x) is the Kronecker product. Church ships
;; multivariate-gaussian and a small set of list / arithmetic
;; primitives; the helpers below build vec / Kronecker / reshape on
;; top of those so the sampled matrix shape matches the QVR
;; declaration. Matrices are represented as Scheme lists of row lists
;; (row-major) so (car m) is the top row.
(define (vec mat)
  ;; Row-major flatten: (vec '((a b) (c d))) -> '(a b c d).
  (apply append mat))

(define (reshape lst p n)
  ;; Inverse of vec on a p x n row-major matrix.
  (if (= p 0)
      '()
      (cons (take lst n) (reshape (drop lst n) (- p 1) n))))

(define (take lst k)
  (if (= k 0) '() (cons (car lst) (take (cdr lst) (- k 1)))))

(define (drop lst k)
  (if (= k 0) lst (drop (cdr lst) (- k 1))))

(define (scale-row r s)
  (map (lambda (x) (* x s)) r))

(define (scale-mat m s)
  (map (lambda (r) (scale-row r s)) m))

(define (mat-kron a b)
  ;; Kronecker product. (a is m x n, b is p x q) -> (mp x nq).
  ;; For each row of a, produce p rows where row i is the row-wise
  ;; concatenation of (b[i] scaled by each a[*][j]).
  (apply append
         (map (lambda (a-row)
                (map (lambda (b-row)
                       (apply append
                              (map (lambda (a-ij)
                                     (scale-row b-row a-ij))
                                   a-row)))
                     b))
              a)))

(define (matrix-normal mu u v)
  ;; vec(X) ~ MultiNormal(vec(mu), V (x) U); reshape the sampled
  ;; vector back to a p x n matrix matching mu's layout.
  (reshape (multivariate-gaussian (vec mu) (mat-kron v u))
           (length mu)
           (length (car mu))))
