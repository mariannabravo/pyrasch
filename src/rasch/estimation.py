import numpy as np

def sigma(z):
    """
    Numerically stable logistic function.

    Parameters
    ----------
    z : float or np.ndarray
        Input value(s).

    Returns
    -------
    float or np.ndarray
        Logistic transformation of z.
    """
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def p_correct(a_s, d_q):
    """
    Calculate the probability of a correct response
    under the dichotomous Rasch model.

    Parameters
    ----------
    a_s : float or np.ndarray
        Person ability parameter(s).

    d_q : float or np.ndarray
        Item difficulty parameter(s).

    Returns
    -------
    float or np.ndarray
        Probability of a correct response.
    """
    return sigma(a_s - d_q)


def log_likelihood(x_sq, a_s, d_q):
    """
    Calculate the Rasch log-likelihood.

    Missing responses coded as np.nan do not
    contribute to the likelihood.

    Parameters
    ----------
    x_sq : np.ndarray
        Response matrix containing 0, 1 and np.nan values.

    a_s : np.ndarray
        Person ability estimates.

    d_q : np.ndarray
        Item difficulty estimates.

    Returns
    -------
    float
        Log-likelihood of the observed response matrix.
    """
    # Mask, False assigned to missing elements
    m_sq = np.logical_not(np.isnan(x_sq)) 
    
    a_minus_d = a_s[:, None] - d_q[None, :]

    # Replace missing responses with zero for the arithmetic to work
    x_observed = np.where(m_sq, x_sq, 0)

    terms = x_observed*a_minus_d - np.logaddexp(0, a_minus_d)

    # Sum only over observed responses
    return np.where(m_sq, terms, 0).sum()




