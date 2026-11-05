import numpy as np

from .estimation import sigma
from .estimation import log_likelihood

from .results import RaschResults


def maximising_f(f, xmin = 0, xmax = 0.5, maxiter = 100, tol = 1e-8, max_x = 16):
    """
    Maximise a 1D function.

    The search interval is expanded if necessary
    and then a five-point search is used to find 
    the maximum.

    Parameters
    ----------
    f : callable
        Function to be maximised.

    xmin : float, default = 0
        Lower search boundary.

    xmax : float, default = 0.5
        Initial upper search boundary.

    maxiter : int, default = 100
        Maximum number of search iterations.

    tol : float, default = 1e-8
        Convergence tolerance for the search interval.

    max_x : float, default = 16
        Largest permitted search value.

    Returns
    -------
    float
        Estimated location of the maximum.
    """
    # Maximise likelihood in 1 dimension to find the best step size to use that iteration
    
    a = xmin 
    b = xmax

    if a >= b:
        raise ValueError("xmin must be smaller than xmax")

    # Track best point encountered
    best_x = a
    best_value = f(a)

    # Expand search interval while f is increasing towards the right boundary
    while b < max_x:
        m = (a + b)/2

        f_m = f(m)
        f_b = f(b)

        if f_m > best_value:
            best_value = f_m
            best_x = m

        if f_b > best_value:
            best_value = f_b
            best_x = b

        # Stop expanding once f is no longer increasing at right boundary

        if f_b <= f_m:
            break

        b = min(2*b, max_x)

    for _ in range(maxiter):
        m = (a + b)/2
        l = (a + m)/2
        r = (m + b)/2

        points = np.array([a, l, m, r, b])
        values = np.array([f(a), f(l), f(m), f(r), f(b)])

        index = np.argmax(values)

        # Update best point seen so far

        if values[index] > best_value:
            best_value = values[index]
            best_x = points[index]

        if index in (0, 1):
            b = m 


        elif index in (3, 4):
            a = m

        else: # index = 2
            a = l
            b = r

        if abs(b - a) < tol:
            break

    return best_x



def fit_rasch(x_sq, tol = 1e-5, gamma_a_tol = 1e-2, gamma_d_tol = 1e-3, max_iterations = 10000):

    """
    Fit the dichotomous Rasch model to the input response matrix.

    Estimation of person and item parameters is performed
    using gradient ascent with adaptive step
    sizes for person ability and item difficulty.

    Missing responses must be coded as np.nan.

    Parameters
    ----------
    x_sq : np.ndarray
        Response matrix containing 0, 1 and np.nan values.

    tol : float, default = 1e-5
        Convergence tolerance for the log-likelihood.

    gamma_a_tol : float, default = 1e-2
        Convergence tolerance for ability step sizes.

    gamma_d_tol : float, default = 1e-3
        Convergence tolerance for difficulty step sizes.

    max_iterations : int, default = 10000
        Maximum number of optimisation iterations.

    Returns
    -------
    RaschResults
        Estimated person abilities, item difficulties and fitting diagnostics.
    """

    x_sq = np.asarray(x_sq, dtype = float)

    if x_sq.ndim != 2:
        raise ValueError("x_sq must be a 2D response matrix")

    m_sq = np.logical_not(np.isnan(x_sq))

    if not np.isin(x_sq[m_sq], [0, 1]).all():
        raise ValueError("Observed responses must be stored as 0 or 1")

    if np.any(m_sq.sum(axis = 0) == 0):
        raise ValueError("At least one item has no observed responses")
    
    if np.any(m_sq.sum(axis = 1) == 0):
        raise ValueError("At least one person has no observed responses")

    no_students, no_items = x_sq.shape

    # Sufficient statistics
    A_s = np.nansum(x_sq, axis = 1)
    D_q = np.nansum(x_sq, axis = 0)

    # Initial parameter estimates
    a_s = np.zeros(no_students)
    d_q = np.zeros(no_items)

    gamma_a = None
    gamma_d = None

    gamma_converged = False

    log_likelihoods = []
    gamma_a_history = []
    gamma_d_history = []

    converged = False
    message = "Maximum number of iterations reached"

    for i in range(max_iterations):
        P_sq = sigma(a_s[:, None] - d_q[None, :])

        # Expected scores from observed responses
        expected_A_s = (m_sq*P_sq).sum(axis = 1)
        expected_D_q = (m_sq*P_sq).sum(axis = 0)

        dl_da = A_s - expected_A_s
        dl_dd = expected_D_q - D_q

        if not gamma_converged:
            likelihood_a = lambda n: log_likelihood(x_sq, a_s + (n*dl_da), d_q)
            likelihood_d = lambda n: log_likelihood(x_sq, a_s, d_q + (n*dl_dd))
        
            new_gamma_a = maximising_f(likelihood_a, 0, 0.5)
            new_gamma_d = maximising_f(likelihood_d, 0, 0.5)
        
            if gamma_a is not None and gamma_d is not None:
        
                if abs(new_gamma_a - gamma_a) < gamma_a_tol and abs(new_gamma_d - gamma_d) < gamma_d_tol:
                    gamma_converged = True
        
            gamma_a = new_gamma_a
            gamma_d = new_gamma_d

        gamma_a_history.append(gamma_a)
        gamma_d_history.append(gamma_d)

        old_value = log_likelihood(x_sq, a_s, d_q)

        a_s += gamma_a * dl_da
        d_q += gamma_d * dl_dd
                
        mean_d = d_q.mean()
        d_q -= mean_d
        a_s -= mean_d
                    
        value = log_likelihood(x_sq, a_s, d_q)

        if value < old_value:
            print(f"Warning: likelihood decreased from {old_value:.4f} to {value:.4f} at iteration {i}")

        log_likelihoods.append(value)
            
        if len(log_likelihoods) > 1 and abs(value - log_likelihoods[-2]) < tol:
            converged = True 
            message = ("Log-likelihood has converged")
            break


    return RaschResults(
        a_s = a_s,
        d_q = d_q,
        log_likelihoods = log_likelihoods,
        gamma_a_history = gamma_a_history,
        gamma_d_history = gamma_d_history,
        converged = converged,
        iterations = len(log_likelihoods),
        message = message,
    )

