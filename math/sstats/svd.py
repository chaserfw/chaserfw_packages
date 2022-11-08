import numpy as np
from sutils import trange

def SVD(n_poi:int, wise_op_mean:np.ndarray, mean_total:np.array, trace_length, n_classes=256):
    """Computes singular value decomposition

    :param n_poi:
    :type n_poi: :class: `int`
    
    :param wise_op_mean:
    :type wise_op_mean: :class: `np.ndarray`
    
    :param mean_total:
    :type mean_total: :class: `np.array`
    
    :param trace_length:
    :type trace_length: :class: `int`

    :param n_classes:
    :type n_classes: :class: `int` default is 256
    
    :return:
    :rtype: :class:`Tuple[projection, U, S, V]`
    """
    B = np.zeros(trace_length)
    for i in trange(n_classes, desc='[INFO]: Computing B matrix'):
        trace   = wise_op_mean[i] - mean_total
        t_trace = np.transpose([trace])
        B       = B + trace * t_trace
    
    B      = B/n_classes
    U,S,V  = np.linalg.svd(B)
    
    projection = U[:,:n_poi]

    return projection, U, S, V