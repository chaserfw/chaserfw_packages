from scipy.special import digamma
from scipy.special import gamma
from annoy import AnnoyIndex
import numpy as np
import scipy.spatial as ss
import numpy.random as nr
from sfileengine import FileEngine
from tqdm import trange
import tables

def generate_forest_from_data(data_file, forest_file, n_trees=10, samples=1000, data_length=2000):
	h5_stream = tables.open_file(data_file, mode='r')
	forest_index = AnnoyIndex(data_length, 'euclidean')

	for i in trange(samples, desc='[INFO]: Adding samples to the forest'):
		v = h5_stream.root.vectors[i]
		forest_index.add_item(i, v)
	
	forest_index.build(n_trees)
	forest_index.save(forest_file)
	h5_stream.close()

def KLdivergence(x, y):
	"""Compute the Kullback-Leibler divergence between two multivariate samples
	Args:
		x : 2D array (n, d)
			Samples from distribution P, which typically represents the true
			distribution.

		y : 2D array (m,d)
			Samples from distribution Q, which typically represents 
			the approximate distribution.
	Returns:
		out: float
			The estimated Kullback-Liebler divergence D(P||Q)
	"""

	n, d  = x.shape
	m, dy = y.shape

	assert(d == dy)

	# Build a K-Tree representation of the samples and find the nearest neighbour
	# of each point in x

	xtree = ss.cKDTree(x)
	ytree = ss.cKDTree(y)


	r = xtree.query(x, k=2, eps=0.01, p=2)[0][:,1]
	s = ytree.query(x, k=1, eps=0.01, p=2)[0]

	return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def KLdivergence_from_forest(x, y, x_forest_index:AnnoyIndex, y_forest_index:AnnoyIndex):
	# Get the first two nearest neighbours for x, since the closest one is the
	# sample itself.
	r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
	s = ytree.query(x, k=1, eps=.01, p=2)[0]

	# There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
	# on the first term of the right hand side.
	return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def knn_based_entropy(x, k=3, base=np.exp(1), intens=1e-10):
	"""The classic K-L k-nearest neighbor continuous entropy estimator
	x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
	if x is a one-dimensional scalar and we have four samples
	NOTA: Parece ser aplicable solo en el caso continuo
	"""
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	d = len(x[0])
	N = len(x)
	print ('N y d', N, d)
	x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
	tree = ss.cKDTree(x) # c de continous?
	#nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
	nn = [tree.query(point, k+1, p=2)[0][k] for point in x]
	#print (nn)
	const = digamma(N)-digamma(k) + d*np.log(2)
	c = list(map(np.log, nn))
	return (const + d*np.mean(c))/np.log(base), tree


def compute_forest_index(fengine:FileEngine, forest_index:AnnoyIndex, k=3, base=np.exp(1), intens=1e-10):
	pass

def ann_based_entropy(forest_index:AnnoyIndex, k=3, base=np.exp(1)):
	"""Compute the entropy using aproximation nearest neighbor (ANN).
	the function assumes the tree is already defined, so that, the intens term is not
	defined.
	"""
	if forest_index.get_n_items() <= k:
		print ('[ERROR]: There are more traces than requested neighbor')
		return None
	
	# Get the lenght of the vectors through a single vector 
	# (all vectors have the same dimension)
	d = len(forest_index.get_item_vector(0))

	# Get total vectors in the index
	N = forest_index.get_n_items()
	print ('N y d', N, d)
	nearest_n = np.empty(shape=N, dtype=np.float)
	for i in trange(N, desc='[INFO *EntropyANN*]: Finding the {}-NN of all vectors in the space'.format(k)):
		nearest_n[i] = np.log(forest_index.get_nns_by_item(1, k+1, include_distances=True)[1][k])

	const = digamma(N) - digamma(k) + d * np.log(2)
	return (const + d * np.mean(nearest_n)) / np.log(base)


def avgdigamma(points, dvec):
	#This part finds number of neighbors in some radius in the marginal space
	#returns expectation value of <psi(nx)>
	N = len(points)
	tree = ss.cKDTree(points)
	avg = 0.
	for i in range(N):
		dist = dvec[i]
		#subtlety, we don't include the boundary point, 
		#but we are implicitly adding 1 to kraskov def bc center point is included
		num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
		avg += digamma(num_points)/N
	return avg

def knn_based_MI_LNC(X, k=5, base=np.exp(1), alpha=0.25, intens=1e-10):

	# N is the number of samples
	N = len(X[0])
	
	# First Step: calculate the mutual information using the Kraskov 
	# mutual information estimator
	
	# Adding small noise to X, e.g., x<-X+noise
	x = []
	for i in range(len(X)):
		tem = []
		for j in range(len(X[i])):
			tem.append([X[i][j] + intens*nr.rand(1)[0]])
		x.append(tem)

	points = []
	for j in range(len(x[0])):
		tem = []
		for i in range(len(x)):	
			tem.append(x[i][j][0])
		points.append(tem)
	
	tree = ss.cKDTree(points)
	dvec = []
	for i in range(len(x)):
		dvec.append([])	
	for point in points:
		#Find k-nearest neighbors in joint space, p=inf means max norm
		knn = tree.query(point, k+1, p=float('inf'))
		points_knn = []
		for i in range(len(x)):
			dvec[i].append(float('-inf'))
			points_knn.append([])
		for j in range(k+1):
			for i in range(len(x)):
				points_knn[i].append(points[knn[1][j]][i])
		
		#Find distances to k-nearest neighbors in each marginal space
		for i in range(k+1):
			for j in range(len(x)):
				if dvec[j][-1] < np.fabs(points_knn[j][i]-points_knn[j][0]):
					dvec[j][-1] =  np.fabs(points_knn[j][i]-points_knn[j][0])

	ret = 0.
	for i in range(len(x)):
		ret -= avgdigamma(x[i],dvec[i])
	ret += digamma(k) - (float(len(x))-1.)/float(k) + (float(len(x))-1.) * digamma(len(x[0]))

def MI_LNC(forest_index:AnnoyIndex, k=5, base=np.exp(1), alpha=0.25, intens=1e-10):


	# N is the number of samples
	N = forest_index.get_n_items()
	
	# First Step: calculate the mutual information using the Kraskov 
	# mutual information estimator
	# adding small noise to X, e.g., x<-X+noise
	x = []
	for i in range(len(X)):
		tem = []
		for j in range(len(X[i])):
			tem.append([X[i][j] + intens*nr.rand(1)[0]])
		x.append(tem)

	points = [];
	for j in range(len(x[0])):
		tem = [];
		for i in range(len(x)):	
			tem.append(x[i][j][0]);
		points.append(tem);
	
	tree = ss.cKDTree(points);
	dvec = [];
	for i in range(len(x)):
		dvec.append([])	
	for point in points:
		#Find k-nearest neighbors in joint space, p=inf means max norm
		knn = tree.query(point,k+1,p=float('inf'));
		points_knn = [];
		for i in range(len(x)):
			dvec[i].append(float('-inf'));
			points_knn.append([]);
		for j in range(k+1):
			for i in range(len(x)):
				points_knn[i].append(points[knn[1][j]][i]);
		
		#Find distances to k-nearest neighbors in each marginal space
		for i in range(k+1):
			for j in range(len(x)):
				if dvec[j][-1] < fabs(points_knn[j][i]-points_knn[j][0]):
					dvec[j][-1] =  fabs(points_knn[j][i]-points_knn[j][0]);

	ret = 0.
	for i in range(len(x)):
		ret -= MI.avgdigamma(x[i],dvec[i]);
	ret += digamma(k) - (float(len(x))-1.)/float(k) + (float(len(x))-1.) * digamma(len(x[0]));
