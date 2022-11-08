import tensorflow as tf


# Gradient Visualisation (Masure et. al. 2018) 
def GradVis(model, X, Y, n_classes=256):
	'''
	Tomado de "Gradient Visualization for General Characterization in Profiling Attacks",
	Masure et. al., 2018
	_________________________________________
	Computar los gradients de la funcion loss de `model` (con respecto a los input features) 
	para cada ejemplo en `X`, usando los labels en `Y`.
	_________________________________________
	Input: 
	model (tensorflow.Model) : Model
	X (numpy.ndarray(batch_size,n_features)): inputs para model
	Y (numpy.ndarray(batch_size)): los labels de los outputs de model
	n_classes (int)=None: Numero de labels distintos en 
	Output:
	grads (np.ndarray(batch_size,n_features,1)): los gradients de la funcion loss (con respecto a los input features)
	'''
	# Crea categorias de clasificacion
	y_true     = tf.keras.utils.to_categorical(Y,num_classes=n_classes)
	
	# Define el tamaño del batch, no entiendo porque
	batch_size = 1 if len(X.shape) == 1 else X.shape[0] 
	
	# Fija los valores None al 1 en el shape del model.input
	# Crea una version nueva Reshaped_X (reshapiando) el conjunto de trazas X
	Reshaped_X = X.reshape(tuple(i or batch_size for i in model.input_shape))
	
	# 
	v          = tf.Variable(Reshaped_X, dtype=model.input.dtype)
	
	# tf.GradientTape hace una grabacion de las operaciones 
	with tf.GradientTape(watch_accessed_variables = False) as tape:
		tape.watch(v)
		losses=model.loss(y_true, model(v))
	grads = tape.gradient(losses, v)
	return grads
#############################
