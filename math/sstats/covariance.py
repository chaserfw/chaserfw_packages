import numpy as np
from sutils import trange

def calc_cov(x, y):
	return np.cov(x, y)