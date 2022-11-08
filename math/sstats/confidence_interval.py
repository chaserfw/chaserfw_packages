import numpy as np
import scipy.stats as st

def ge_confidence_mean(ge_matrix, conf=0.95):
    """Returns confidence interval of mean

    Args:
        ge_matrix (2D numpy): A guessing entropy matrix 
        conf (float): Confident' interval (default is 0.95)

    Returns:
        A tuple with the interval represented as the bound of the guessing entropy
    """
    # Compute the mean, the error, and the inverse of the CDF.
    # A normal distribution is assumed
    mean, sem, m = np.mean(ge_matrix, axis=0), st.sem(ge_matrix), st.t.ppf((1+conf)/2., len(ge_matrix)-1)
    return mean - m*sem, mean + m*sem