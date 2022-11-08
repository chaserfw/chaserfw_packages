import tensorflow as tf
import numpy as np
from sutils import trange

def CNN_Heatmap(model, dataset, conv_lay_idx=0):
	'''
	Tomado de "Methodology for Efficient CNN Architectures in Profiling Attacks",
	Zaid et. al., 2019
	_________________________________________
	Visualizar la media, sobre los filters, de cada output del ith (i=`conv_lay_idx`) layer Conv1D  de `model` 
	para cada ejemplo en `X`.
	_________________________________________
	Input: 
	model (tensorflow.keras.Model) : Model
	X (numpy.ndarray(batch_size,n_features)): inputs para model
	conv_lay_idx (int)=0: El layer Conv1D deseado (i = el ith layer Conv1D, no el ith layer de todos etc.
	Output:
	heatmap (np.ndarray(batch_size,n_outputs)): las medias de los outputs del layer, sobre los filters
	'''
	from tensorflow.python.ops import nn_ops
	conv_indices = [idx for idx,layer in enumerate(model.layers) if type(layer)==tf.keras.layers.Conv1D]
	if len(conv_indices)==0: raise Exception('Ningún Conv1D layer identificado. Es posible que el NN no es un CNN.')
	
    # from the specified model take the convolutional layer (from the convolutional block)
	layer   = model.get_layer(index=conv_indices[conv_lay_idx])
	f       = tf.keras.Model(inputs=model.inputs,outputs=[layer.input])(dataset)
	df_1d   = 'NWC' if layer.data_format=='channels_last' else 'NCW'
	WT_a    = nn_ops.conv1d(f, layer.kernel, layer.strides, layer.padding.upper(), data_format=df_1d)
	
	#WT_a have the following shape: (batch_size, n_sample_points, n_filters)
	heatmap = np.mean(WT_a, axis=-1) 
	return heatmap

def CNN_Heatmap_comparison(model, dataset, conv_lay_idx=0, dataset_snr=None) -> np.array:
    """From a CNN model, it computes Heatmap and its dataset SNR part

    Retrieves an numpy array, where firts element is the heatmap and the second one
    is the SNR.

    Args:
        model:
            A tensorflow CNN model. CNN model instance.
        dataset:
            A dataset of traces. It should be a numpy array.
        conv_lay_idx:
            The layer from which the heatmap is going to be computed. default (0)
        dataset_snr:
            SNR (mean/std) computed from the dataset. If None, the function computes it

    Returns:
        A two entries numpy array, first element is the heatmap,
        the second element is the SNR:
    """
    if dataset_snr is None:
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        for i in trange(dataset.shape[0], desc='[INFO]: Computing means and varriance'):
            ss.partial_fit(dataset[i])
        dataset_snr = ss.mean_/np.sqrt(ss.var_)
        dataset_snr[np.isnan(dataset_snr)] = 0
    
    hmap = CNN_Heatmap(model, dataset, conv_lay_idx=conv_lay_idx)
    m_hmap = np.mean(np.abs(hmap), axis=0)

    return np.array([m_hmap, dataset_snr], dtype=np.float)
        
