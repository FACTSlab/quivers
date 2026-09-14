;; QIEC distribution bridge for generated Church programs. A QIEC tensor is
;; a tuple record until it reaches a distribution, where it becomes a
;; plain nested list; a log density is the runtime's `dist-score` at the
;; converted value, Booleans scoring as the numbers `flip` expects.
(define (_qvr-qiec-array value)
  (let ((value (_qvr-qiec-value value)))
    (cond
      ((and (pair? value) (equal? (_qvr-qiec-get value "qiec" #f) "tuple"))
       (map _qvr-qiec-array (_qvr-qiec-get value "items" '())))
      ((list? value) (map _qvr-qiec-array value))
      ((eq? value #t) 1)
      ((eq? value #f) 0)
      (else value))))
(define (_qvr-qiec-log-density distribution value)
  (dist-score distribution (_qvr-qiec-array value)))
