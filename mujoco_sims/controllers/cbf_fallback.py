import numpy as np
import osqp
import scipy.sparse as sparse


def solve_least_violation(acc_ref, A, lower, upper, n_bounds, weight=1e3, max_iter=100):
    """
    Input for a CBF-QP that is primal infeasible, i.e. whose CBF constraints
    cannot be satisfied together within the input bounds: the input that
    minimises the largest violation t of the CBF constraints, traded off with
    the tracking of acc_ref,

        min  0.5 ||u - acc_ref||^2 + weight t
        s.t. the first n_bounds rows of A (input bounds), as in the CBF-QP,
             the other rows of A relaxed by t,  t >= 0

    The relaxed problem is always feasible. Returns None if OSQP does not
    return a solution.
    """
    A = sparse.csc_matrix(A)
    n_inputs = A.shape[1]
    n_cbf = A.shape[0] - n_bounds

    # Column of t: zero in the input bounds, one in the CBF constraints
    relaxation = sparse.csc_matrix(np.r_[np.zeros(n_bounds), np.ones(n_cbf)][:, None])
    A_relaxed = sparse.bmat(
        [
            [A, relaxation],
            [None, sparse.csc_matrix([[1.0]])],
        ],
        format="csc",
    )

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.diags(np.r_[np.ones(n_inputs), 0.0], format="csc"),
        q=np.r_[-np.asarray(acc_ref, dtype=float), weight],
        A=A_relaxed,
        l=np.r_[lower, 0.0],
        u=np.r_[upper, np.inf],
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=max_iter,
    )

    results = solver.solve()

    if results.x is None:
        return None

    return np.asarray(results.x[:n_inputs], dtype=float)
