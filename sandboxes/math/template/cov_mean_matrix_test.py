from templates import cov_mean_matrix
from templates import compute_cov_mean_matrix
from sfileengine import H5FileEngine
import numpy as np
from sutils import trange

mode = 'r'
if mode == 'r':
    covmatrix2 = np.load('cov_mean_matrix_2_covmatrix.npy')
    covmatrix = np.load('cov_mean_matrix_covmatrix.npy')
    print (covmatrix2[0].shape)
    print ('-------')
    print (covmatrix[0].shape)
    print ('-------------------')
    print (covmatrix2[0])
    print ('-------')
    print (covmatrix[0])
    for i in trange(covmatrix.shape[0]):
        assert (covmatrix[i] == covmatrix2[i]).all() and covmatrix[i].shape == covmatrix2[i].shape
    print ('All asserts')
else:
    dataset_path = 'ASCAD-compress.h5'
    file = H5FileEngine(dataset_path, group='/Profiling_traces')

    mean, cov_matrix = compute_cov_mean_matrix(file)
    print (mean.shape)
    print (cov_matrix.shape)
    np.save('cov_mean_matrix_2_mean', mean)
    np.save('cov_mean_matrix_2_covmatrix', cov_matrix)

    mean, cov_matrix = cov_mean_matrix(file)
    file.close()
    print (mean.shape)
    print (cov_matrix.shape)

    np.save('cov_mean_matrix_mean', mean)
    np.save('cov_mean_matrix_covmatrix', cov_matrix)