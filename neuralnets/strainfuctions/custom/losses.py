import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

#===============================================================================
def loss_sca(alpha_value=10, nb_class=256):

	# Rank loss function
	def ranking_loss_sca(y_true, y_pred):
		# y_true is not use, it can be dismissed by the equation

		alpha = tf.keras.backend.constant(alpha_value, dtype='float32')
		# Predictor output shape [batch_size, 512].
		# Remove from the batch the firts 256 index which are not valid for 
		# the loss function; however, it this 256 firts index used for the GE 
		# when attacking. When using along with GE on the training 
		# it require to do nothing with regard to y_true
		y_pred = y_pred[:, nb_class:]
		
		# Batch_size initialization
		y_true_int = tf.keras.backend.cast(y_pred, dtype='int32')
		batch_s = tf.keras.backend.cast(tf.keras.backend.shape(y_true_int)[0], dtype='int32')

		# Indexing the training set (range_value = (?,))
		range_value = tf.keras.backend.arange(0, batch_s, dtype='int64')

		# Get rank and scores associated with the secret key (rank_sk = (?,))
		values_topk_logits, indices_topk_logits = tf.nn.top_k(y_pred, k=nb_class, sorted=True) # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
		rank_sk = tf.where(tf.equal(tf.keras.backend.cast(indices_topk_logits, dtype='int64'), tf.reshape(tf.keras.backend.argmax(y_true_int), [tf.shape(tf.keras.backend.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape(?,))
		score_sk = tf.gather_nd(values_topk_logits, tf.keras.backend.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), tf.reshape(rank_sk-1, [tf.shape(rank_sk)[0], 1])])) # Score of the secret key (shape(?,))

		# Ranking Loss Initialization
		loss_rank = 0

		for i in range(nb_class):

			# Score for each key hypothesis (s_i_shape=(?,))
			s_i = tf.gather_nd(values_topk_logits, tf.keras.backend.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), i*tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

			# Indicator function identifying when (i == secret key)
			indicator_function = tf.ones(batch_s) - (tf.keras.backend.cast(tf.keras.backend.equal(rank_sk-1, i), dtype='float32') * tf.ones(batch_s))

			# Logistic loss computation
			logistic_loss = tf.keras.backend.log(1 + tf.keras.backend.exp(- alpha * (score_sk - s_i)))/tf.keras.backend.log(2.0)

			# Ranking Loss computation
			loss_rank = tf.reduce_sum((indicator_function * logistic_loss))+loss_rank

		return loss_rank/(tf.keras.backend.cast(batch_s, dtype='float32'))

	#Return the ranking loss function
	return ranking_loss_sca
#===============================================================================
class LossSCA(tf.keras.losses.Loss):
	def __init__(self, alpha_value=10, nb_class=256):
		super().__init__()
		self.alpha_value  = alpha_value
		self.nb_class     = nb_class

	# Rank loss function
	def call(self, y_true, y_pred):
		alpha = K.constant(self.alpha_value, dtype='float32')
		y_pred = y_pred[:, self.nb_class:]
		
		# Batch_size initialization
		y_true_int = K.cast(y_pred, dtype='int32')
		batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')

		# Indexing the training set (range_value = (?,))
		range_value = K.arange(0, batch_s, dtype='int64')

		# Get rank and scores associated with the secret key (rank_sk = (?,))
		values_topk_logits, indices_topk_logits = tf.nn.top_k(y_pred, k=self.nb_class, sorted=True) # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
		rank_sk = tf.where(tf.equal(K.cast(indices_topk_logits, dtype='int64'), tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape(?,))
		score_sk = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), tf.reshape(rank_sk-1, [tf.shape(rank_sk)[0], 1])])) # Score of the secret key (shape(?,))

		# Ranking Loss Initialization
		loss_rank = 0

		for i in range(self.nb_class):

			# Score for each key hypothesis (s_i_shape=(?,))
			s_i = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), i*tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

			# Indicator function identifying when (i == secret key)
			indicator_function = tf.ones(batch_s) - (K.cast(K.equal(rank_sk-1, i), dtype='float32') * tf.ones(batch_s))

			# Logistic loss computation
			logistic_loss = K.log(1 + K.exp(- alpha * (score_sk - s_i)))/K.log(2.0)

			# Ranking Loss computation
			loss_rank = tf.reduce_sum((indicator_function * logistic_loss))+loss_rank
		
		loss_rank = loss_rank/(K.cast(batch_s, dtype=tf.float32))
		
		return loss_rank
#===============================================================================
class EpochCountDownWrapperLoss(tf.keras.losses.Loss):
	def __init__(self, loss_fn, epoch_couter_cb, trigger_value):
		super().__init__()
		self.loss_fn = loss_fn
		self.epoch_couter_cb = epoch_couter_cb
		self.trigger_value = trigger_value

	def call(self, y_true, y_pred):
		loss_rank = self.loss_fn(y_true, y_pred)
		return K.switch(self.epoch_couter_cb.get_counter() >= self.trigger_value,
				loss_rank, K.zeros_like(loss_rank))
#===============================================================================
class GECatCrossentropy:
	def __init__(self, classes=256, la_threshold=0):
		self.classes = classes
		self.la_threshold = la_threshold
		self.counter = 0
		
	def loss_function(self, y_true, y_pred):
		if self.counter >= self.la_threshold:
			return tf.keras.losses.categorical_crossentropy(y_true[:, :self.classes], y_pred)
		else:
			self.counter += 1
			return tf.constant(0)
			
class ConstrativeLoss(tf.keras.losses.Loss):
	def __init__(self, margin=None):
		super().__init__(name='ConstrativeLoss')
		self.__margin = margin
	
	def call(self, y_true, y_pred):
		square_pred = tf.keras.backend.square(y_pred)
		margin_square = tf.keras.backend.square(tf.keras.backend.maximum(self.__margin - y_pred, 0))
		return tf.keras.backend.mean(y_true * square_pred + (1-y_true) * margin_square)

class ConstrativeMeanSquareLoss(tf.keras.losses.MeanSquaredError):
	def __init__(self, margin=None):
		super().__init__(reduction=tf.keras.losses_utils.ReductionV2.AUTO, name='mean_squared_error')
		self.__margin = margin
	
	def call(self, y_true, y_pred):
		square_pred = tf.keras.backend.square(y_pred)
		margin_square = tf.keras.backend.square(tf.keras.backend.maximum(self.__margin - y_pred, 0))
		return tf.keras.backend.mean(y_true * square_pred + (1-y_true) * margin_square)
		
#===============================================================================
def define_triplet_loss(margin=0.2, squared=False, type_of_triplets=1):
	'''1. hard_triplets, 2. all_triplets
	'''
	def all_triplet_loss(y_true, y_pred):
		# y_true is not used
		#del y_true
		batch_size = tf.shape(y_pred)[0]
		labels     = tf.cast(y_pred[:, :1, :], dtype=tf.int8)
		embeddings = tf.squeeze(y_pred[:, 1:, :], axis=2)

		dot_prod      = tf.matmul(embeddings, tf.transpose(embeddings))
		square_norm   = tf.linalg.diag_part(dot_prod)
		pairwise_dist = tf.expand_dims(square_norm, 1) - 2.0 * dot_prod + tf.expand_dims(square_norm, 0)
		pairwise_dist = tf.maximum(pairwise_dist, 0.0)
		
		if not squared:
			mask = tf.cast(tf.equal(pairwise_dist, 0.0), dtype=tf.float32)
			pairwise_dist = pairwise_dist + mask * tf.keras.backend.epsilon()
			pairwise_dist = tf.sqrt(pairwise_dist)
			pairwise_dist = pairwise_dist * (1.0 - mask)
		
		anchor_pos_dist = tf.expand_dims(pairwise_dist, 2)
		anchor_neg_dist = tf.expand_dims(pairwise_dist, 2)
		
		triplet_loss = anchor_pos_dist - anchor_neg_dist + margin
		
		# Getting mask for all valid triplets
		indices_eq  = tf.cast(tf.eye(batch_size), dtype=tf.bool)
		indices_neq = tf.logical_not(indices_eq)
		i_nq_j      = tf.expand_dims(indices_neq, 2)
		i_nq_k      = tf.expand_dims(indices_neq, 1)
		j_nq_k      = tf.expand_dims(indices_neq, 0)
		
		temporal_i_nq_j = tf.logical_and(i_nq_j, i_nq_k)
		distinct_indices = tf.logical_and(temporal_i_nq_j, j_nq_k)
		
		label_q = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
		i_q_j   = tf.expand_dims(label_q, 2)
		i_q_k   = tf.expand_dims(label_q, 1)
		not_i_q_k = tf.logical_not(i_q_k)
		valid_labels = tf.logical_and(i_q_j, not_i_q_k)
		
		mask = tf.cast(tf.logical_and(distinct_indices, valid_labels), dtype=tf.float32)
		
		# Applying mask
		triplet_loss = tf.multiply(mask, triplet_loss)
		triplet_loss = tf.maximum(triplet_loss, 0.0)
		
		valid_triplets     = tf.cast(tf.greater(triplet_loss, tf.keras.backend.epsilon()), dtype=tf.float32)
		num_pos_triplets   = tf.reduce_sum(valid_triplets)
		
		# lo siguiente puede ser una metrica
		#num_valid_triplets = tf.reduce_sum(mask)
		#fraction_pos_triplets = num_pos_triplets / (num_valid_triplets + tf.backend.epsilon())
		
		return tf.reduce_sum(triplet_loss) / (num_pos_triplets + tf.keras.backend.epsilon())
	
	return all_triplet_loss
#===============================================================================
def pairwise_distance(feature, squared=False):
	"""Computes the pairwise distance matrix with numerical stability.

	output[i, j] = || feature[i, :] - feature[j, :] ||_2

	Args:
	  feature: 2-D Tensor of size [number of data, feature dimension].
	  squared: Boolean, whether or not to square the pairwise distances.

	Returns:
	  pairwise_distances: 2-D Tensor of size [number of data, number of data].
	"""
	
	feature = tf.squeeze(feature, axis=2)
	
	pairwise_distances_squared = math_ops.add(
		math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
		math_ops.reduce_sum(
			math_ops.square(array_ops.transpose(feature)),
			axis=[0],
			keepdims=True)) - 2.0 * math_ops.matmul(feature,
													array_ops.transpose(feature))

	# Deal with numerical inaccuracies. Set small negatives to zero.
	pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
	# Get the mask where the zero distances are at.
	error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

	# Optionally take the sqrt.
	if squared:
		pairwise_distances = pairwise_distances_squared
	else:
		pairwise_distances = math_ops.sqrt(
			pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

	# Undo conditionally adding 1e-16.
	pairwise_distances = math_ops.multiply(
		pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

	num_data = array_ops.shape(feature)[0]
	# Explicitly set diagonals to zero.
	mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
		array_ops.ones([num_data]))
	pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
	return pairwise_distances

def masked_maximum(data, mask, dim=1):
	"""Computes the axis wise maximum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the maximum.

	Returns:
	  masked_maximums: N-D `Tensor`.
		The maximized dimension is of size 1 after the operation.
	"""
	axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
	masked_maximums = math_ops.reduce_max(
		math_ops.multiply(data - axis_minimums, mask), dim,
		keepdims=True) + axis_minimums
	return masked_maximums

def masked_minimum(data, mask, dim=1):
	"""Computes the axis wise minimum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the minimum.

	Returns:
	  masked_minimums: N-D `Tensor`.
		The minimized dimension is of size 1 after the operation.
	"""
	axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
	masked_minimums = math_ops.reduce_min(
		math_ops.multiply(data - axis_maximums, mask), dim,
		keepdims=True) + axis_maximums
	return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
	del y_true
	margin = 0.2
	labels = tf.squeeze(y_pred[:, :1], axis=2)

 
	labels = tf.cast(labels, dtype='int32')

	embeddings = y_pred[:, 1:]

	### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
	
	# Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
	# lshape=array_ops.shape(labels)
	# assert lshape.shape == 1
	# labels = array_ops.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = pairwise_distance(embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = math_ops.equal(labels, array_ops.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = math_ops.logical_not(adjacency)

	# global batch_size  
	batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

	# Compute the mask.
	pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
	mask = math_ops.logical_and(array_ops.tile(adjacency_not, [batch_size, 1]), 
								math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1])))
	
	mask_final = array_ops.reshape(
		math_ops.greater(
			math_ops.reduce_sum(
				math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True), 0.0), [batch_size, batch_size])
	
	mask_final = array_ops.transpose(mask_final)

	adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
	mask = math_ops.cast(mask, dtype=dtypes.float32)

	# negatives_outside: smallest D_an where D_an > D_ap.
	negatives_outside = array_ops.reshape(
		masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
	negatives_outside = array_ops.transpose(negatives_outside)

	# negatives_inside: largest D_an.
	negatives_inside = array_ops.tile(
		masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
	semi_hard_negatives = array_ops.where(
		mask_final, negatives_outside, negatives_inside)

	loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

	mask_positives = math_ops.cast(
		adjacency, dtype=dtypes.float32) - array_ops.diag(
		array_ops.ones([batch_size]))

	# In lifted-struct, the authors multiply 0.5 for upper triangular
	#   in semihard, they take all positive pairs except the diagonal.
	num_positives = math_ops.reduce_sum(mask_positives)

	semi_hard_triplet_loss_distance = math_ops.truediv(
		math_ops.reduce_sum(
			math_ops.maximum(
				math_ops.multiply(loss_mat, mask_positives), 0.0)),
		num_positives,
		name='triplet_semihard_loss')
	
	### Code from Tensorflow function semi-hard triplet loss ENDS here.
	return semi_hard_triplet_loss_distance
#===============================================================================