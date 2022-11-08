import numpy as np
from sutils import trange
from sfileengine import FileEngine
from processors import ID_wise_mean
from processors import ID_wise_mean_balanced
# ======================================================================================
#
# ======================================================================================
def covariance_templates(clustered_traces:np.ndarray, n_classes:int, projection:np.ndarray):
	"""Build the mean and variance templates based on reduce train samples

	:param train_data: Sorted grouped array of traces, it normally comes from the output of a 
		sorting functions which sorts and groups the traces by an specific criterion e.g. label
		or Sbox output value.
	:type train_data: np.ndarray

	:param n_classes:
	:type n_classes: int

	:param projection:
	:type projection:
	"""
	__mean = np.empty(0)
	__smat = np.empty(0)
	__n_poi = projection.shape(1)
	for i in trange(n_classes, desc='[INFO *CovarianceTemplates*]: Computing covariance and mean matrixes'):
		__full_projected_train = np.empty(0)

		# It uses the feature space (projection) and projects to it each of the grouped traces.
		# i.e. projected_train is the projected traces of the i-th class
		# By doing this projection, the point of interest are gathered or re-arranged somehow
		# that might help the algorithm in choosing them smartly; from the projection
		# a n_poi of points is selected to build a mean matrix out of them.
		__projected_train = np.dot(clustered_traces[i], projection)
		__full_projected_train = np.append(__full_projected_train, __projected_train)
		__full_projected_train = np.reshape(__full_projected_train, (clustered_traces[i].shape[0], __n_poi))
		
		# Appends the mean of the projected train of the specific trace group
		# Here, the mean matrix is created, notice that this is the mean matrix
		# at each point of interest
		__mean = np.append(__mean, np.mean(__full_projected_train, axis=0))
		
		# After creating the mean matrix, we proceed to create the covariance matrix.
		# Similarly, to the mean matrix of each PoI, the covariance matrix is a matrix
		# formed as M_(n_poixn_poi)
		__smat = np.append(__smat, np.cov(np.transpose(__full_projected_train)))
	
	__cov_matrix = np.reshape(__smat, (n_classes, __n_poi, __n_poi))
	__mean_matrix = np.reshape(__mean, (n_classes, __n_poi))

	return __mean_matrix, __cov_matrix
# ======================================================================================
#
# ======================================================================================
def cov_mean_matrix(fileEngine:FileEngine, n_classes:int=256):
	"""Build the mean and variance templates based on reduce train samples

	:param train_data: Sorted grouped array of traces, it normally comes from the output of a 
		sorting functions which sorts and groups the traces by an specific criterion e.g. label
		or Sbox output value.
	:type train_data: `np.ndarray`

	:param n_classes:
	:type n_classes: int
	"""
	__mean = np.empty((256, fileEngine.TotalSamples))
	__cov_matrix = np.empty((256, fileEngine.TotalSamples, fileEngine.TotalSamples))
	
	_, op_mean, _, clustered_traces = ID_wise_mean(fileEngine, clustered_traces=True)
	__mean[:] = op_mean[:]
	for i_class in trange(n_classes, desc='[INFO *CovarianceTemplates*]: Computing covariance and mean matrices', position=0):
		for i in trange(fileEngine.TotalSamples, desc='Row of {}'.format(i_class), position=1, leave=False):
			for j in trange(fileEngine.TotalSamples, desc='col of {} and {}'.format(i_class, i), position=2, leave=False):
				# Set the next trace
				trace_meta = np.array(fileEngine[clustered_traces[i_class]], dtype=np.ndarray)
				# Get the trace from the smaller file engine
				traces = np.vstack(trace_meta[:,0])
				x = traces[:, i]
				y = traces[:, j]
				__cov_matrix[i_class, i, j] = np.cov(x, y)[0][1]
	
	return __mean, __cov_matrix
# ======================================================================================
#
# ======================================================================================
def _compute_covmean_matrix(n_classes, function_callback, **kwargs):
	fileEngine = kwargs['dataset']

	__mean = np.empty((256, fileEngine.TotalSamples))
	smat = np.empty(0)

	_, op_mean, _, clustered_traces = function_callback(**kwargs)

	__mean[:] = op_mean[:]
	
	for i_class in trange(n_classes, desc='[INFO *CovMeanTemplates*]: Computing covariance and mean matrices', position=0):
		# Set the next trace
		trace_meta = np.array(fileEngine[clustered_traces[i_class]], dtype=np.ndarray)
		# Get the trace from the smaller file engine
		traces = np.vstack(trace_meta[:,0])
		smat = np.append(smat, np.cov(np.transpose(traces)))

	__cov_matrix = np.reshape(smat, (n_classes, fileEngine.TotalSamples, fileEngine.TotalSamples))
	
	return __mean, __cov_matrix
# ======================================================================================
#
# ======================================================================================
def covmean_matrix(
		fileEngine:FileEngine, n_classes:int=256, n_traces:int=None, 
		plaintext_pos:int=None, key_pos:int=None, by_label:bool=True):
	"""Build the mean and variance templates based a TraceSet. The model is based on ID leakage model.

	:param fileEngine: A fileEngine that maps the TraceSet, normally, it represents the train set in a template attack
	:type fileEngine: :class:`FileEngine`

	:param n_classes:
	:type n_classes: int
	"""
	
	m_kwargs = {'dataset':fileEngine, 
		'plaintext_pos':plaintext_pos, 
		'key_pos':key_pos, 
		'n_traces':n_traces, 
		'by_label':by_label, 
		'clustered_traces':True}

	return _compute_covmean_matrix(n_classes, ID_wise_mean, **m_kwargs)

# ======================================================================================
#
# ======================================================================================
def covmean_matrix_balanced(
		fileEngine:FileEngine, n_classes:int=256, n_indexes:int=None, 
		balancer_file=None, plaintext_pos:int=None, key_pos:int=None, 
		by_label:bool=True):
	"""Build the mean and variance templates based a TraceSet. The model is based on ID leakage model.

	:param fileEngine: A fileEngine that maps the TraceSet, normally, it represents the train set in a template attack
	:type fileEngine: :class:`FileEngine`

	:param n_classes:
	:type n_classes: int
	"""
	m_kwargs = {'dataset':fileEngine,
		'balancer_file':balancer_file, 
		'n_indexes':n_indexes,
		'plaintext_pos':plaintext_pos, 
		'key_pos':key_pos, 
		'n_indexes':n_indexes, 
		'by_label':by_label, 
		'clustered_traces':True}
				
	return _compute_covmean_matrix(n_classes, ID_wise_mean_balanced, **m_kwargs)
# ======================================================================================
#
# ======================================================================================
def template_predict(
		test_fileengine:FileEngine, mean_matrix:np.ndarray, 
		cov_matrix:np.ndarray, n_classes:int=256):
	"""
	:param test_fileengine:
	:type test_fileengine: :class:`FileEngine`

	:param mean_matrix:
	:type mean_matrix: `np.ndarray`

	:param cov_matrix:
	:type cov_matrix: `np.ndarray`
	
	:param n_classes:
	:type n_classes: `int`
	"""
	smat_pool = np.sum(cov_matrix, axis=0)/n_classes
	inv_smat_pool = np.linalg.inv(smat_pool)

	res = np.zeros(shape=(test_fileengine.shape[0], n_classes))
	for i in trange(test_fileengine.shape[0], desc='[INFO]: Computing template predictions', position=0):
		for k in range(n_classes):
			T_k      = np.array(test_fileengine[i][0], dtype=float) - mean_matrix[k]
			res[i,k] = -0.5* (np.dot(np.dot(T_k , inv_smat_pool), np.transpose(T_k)))
	
	res     = np.reshape(res,(test_fileengine.shape[0], n_classes))
	predict = np.empty((test_fileengine.shape[0], n_classes))
	print (res)
	for k in range(test_fileengine.shape[0]):
		predict[k] = np.flip(np.argsort(res[k]))
	return predict
# ======================================================================================
#
# ======================================================================================
class CovMeanMatrix:
	@staticmethod
	def from_fileengine():
		pass