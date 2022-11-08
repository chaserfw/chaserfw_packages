import tensorflow as tf
import os

class Encoder():
	def __init__(self, encoder):
		self.__encoder = encoder
	
	@property
	def Layers(self):
		return self.__encoder.layers

class InformationPath:
	'''Autoencoder should be formed as a 3 items list; 
		- its global input.
		- the encoder, as a functional model
		- the decoder, as a functional model


	'''
	def __init__(self, file_engine, autoencoder, encoder_n_samples, add_id=None, training_dir=None, des_scalers_list=None, mtd_descriptor=None, 
	second_file_engine=None, batch_size=None, fengine_total_traces=None, second_fengine_total_traces=None):
		self.__autoencoder = autoencoder
		self.__file_engine = file_engine
		self.__training_dir = training_dir

		training_result_dir = os.path.join('..', 'autoencoder_trainig_result')
		model_id = '20210601-135432'
		model = tf.keras.models.load_model(os.path.join(training_result_dir, model_id, 'epochs_models', 'model_e0.h5'), compile=False)
	
	@property
	def AutoencoderLayers(self):
		return self.__autoencoder.layers
	
	@property
	def get_autoencoder_layer(self, index):
		return self.__autoencoder.get_layer(index)

	@property
	def Encoder(self):
		return self.__autoencoder[1]

	@property
	def Decoder(self):
		return self.__autoencoder[2]

	def generate_output_decoder_model(self, enc_out_index=1, ):
		f     = tf.keras.Model(inputs=self.__autoencoder.layers[1].inputs, 
							   outputs=[self.__autoencoder.layers[1].layers[enc_out_index].output])#(Reshaped_X_validation)
		WT_a = f.predict(Reshaped_X_validation)



print ('ENCODER CONV LAYER 1 (2) - throught the loaded autoencoder')
layer = model.layers[1].get_layer(index=1) #encoder.get_layer(index=1)
print (layer.input)
print (model.layers[1].inputs)
print ('layer name', model.layers[1].layers[5].name)
#loaded_encoder = model.layers[1]
#loaded_decoder = model.layers[2]
#output_loaded_encoder = loaded_encoder(Reshaped_X_validation)


f2     = tf.keras.Model(inputs=model.layers[2].inputs, outputs=[model.layers[2].layers[2].output])#(Reshaped_X_validation)
WT_b = f2.predict(WT_a)

print ('WT_a', WT_a.shape)
print ('WT_b', WT_b.shape)

heatmap=np.mean(WT_a, axis=-1)
print (heatmap.shape)
print (heatmap)

heatmap2=np.mean(WT_b, axis=-1)
print (heatmap2.shape)
print (heatmap2)