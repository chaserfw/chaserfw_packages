from ..trainers import AbstractTrainer
from ..trainfunctions import check_file_exists
from ..trainfunctions import OneCycleLR
import tensorflow as tf
#import tensorflow.keras as tk
import matplotlib
import matplotlib.pyplot as plt
import os
from sutils import check_for_modules

#======================================================================================
'''
Callback flags for trainer's train function
'''
SAVE_MODEL = 0x01
LR_MANAGER = 0x02
NO_PRESET_CALLBACKS = 0x00
#======================================================================================
class TrainerValAE(AbstractTrainer):
	def __init__(self, model, epochs:int=150, batch_size:int=100, 
		train_dir:str='./', maxlr:float=1e-3, sfilename:str=None, 
		model_name:str='autoencoder'):
		
		super().__init__(model, epochs=epochs, batch_size=batch_size, train_dir=train_dir, maxlr=maxlr, sfilename=sfilename)
		self.encoder    = None
		self.decoder    = None
		self.model_name = model_name
		self.__WARNING  = '[WARNING -{}-]:'.format(self.__class__.__name__)
		self.__ERROR    = '[ERROR -{}-]:'.format(self.__class__.__name__)
		self.__INFO     = '[INFO -{}-]:'.format(self.__class__.__name__)
	
	@property
	def Encoder(self):
		return self.encoder
	
	@property
	def Decoder(self):
		return self.decoder
	
	@Encoder.setter
	def Encoder(self, encoder):
		self.encoder = encoder
	
	@Decoder.setter
	def Decoder(self, decoder):
		self.decoder = decoder
	
	def PlotModel(self):
		if not check_for_modules({'pydot'}):
			tf.keras.utils.plot_model(self.Model, to_file=os.path.join(self.TrainDir, '{}.png'.format(self.model_name)), show_shapes=True)
			if self.Encoder is not None:
				tf.keras.utils.plot_model(self.Encoder, to_file=os.path.join(self.TrainDir, 'encoder.png'), show_shapes=True)
			if self.Decoder is not None:
				tf.keras.utils.plot_model(self.Decoder, to_file=os.path.join(self.TrainDir, 'decoder.png'), show_shapes=True)
		else:
			print (self.__WARNING, 'Some package required to plot the model are missing, model does not be plotted.')
		
	def SaveModel(self):
		with open(os.path.join(self.TrainDir, 'AE_summary.txt'), 'w') as summary:
			self.Model.summary(print_fn=lambda x: summary.write(x + '\n'))
		model_json = self.Model.to_json()
		with open(os.path.join(self.train_dir, 'model.json'), "w") as json_file:
			json_file.write(model_json)
		
		if self.Encoder is not None:
			with open(os.path.join(self.TrainDir, 'ENC_summary.txt'), 'w') as summary:
				self.Encoder.summary(print_fn=lambda x: summary.write(x + '\n'))
			model_json = self.Encoder.to_json()
			with open(os.path.join(self.train_dir, 'encoder.json'), "w") as json_file:
				json_file.write(model_json)
			
			self.Encoder.save(os.path.join(self.train_dir, 'encoder.h5'))
		
		if self.Decoder is not None:
			with open(os.path.join(self.TrainDir, 'DEC_summary.txt'), 'w') as summary:
				self.Decoder.summary(print_fn=lambda x: summary.write(x + '\n'))
			model_json = self.Decoder.to_json()
			with open(os.path.join(self.train_dir, 'decoder.json'), "w") as json_file:
				json_file.write(model_json)

			self.Decoder.save(os.path.join(self.train_dir, 'decoder.h5'))
	
	def train(self, savemodel=True, preset_callbacks=SAVE_MODEL | LR_MANAGER, **kwargs):
		file_name = os.path.join(self.train_dir, self.sfilename)
		check_file_exists(os.path.dirname(file_name))
		
		all_callbacks = None if not bool(preset_callbacks & ~NO_PRESET_CALLBACKS) else []

		if preset_callbacks & LR_MANAGER:
			all_callbacks.append(OneCycleLR(max_lr=self.MaxLR, end_percentage=0.2, 
									scale_percentage=0.1, maximum_momentum=None, 
									minimum_momentum=None,verbose=True, 
									b_size_sent=self.batch_size, sps=self.OCP_sps))

		# Save model every epoch
		if preset_callbacks & SAVE_MODEL:
			all_callbacks.append(tf.keras.callbacks.ModelCheckpoint(file_name))
		
		all_callbacks.extend(self.callbacks_list)

		print (self.__INFO, 'Using {}'.format(self.__class__.__name__))
		
		history = self.Model.fit(x=self.x_train_dict, y=self.y_train_dict, 
					validation_data=(self.x_val_dict, self.y_val_dict) if self.x_val_dict is not None and self.y_val_dict is not None else None, 
					batch_size=self.batch_size, epochs=self.epochs,
					callbacks=all_callbacks, **kwargs)
		
		if savemodel:
			print (self.__INFO, 'Saving models')
			self.SaveModel()
			print (self.__INFO, 'Plotting models')
			self.PlotModel()
				
		print (self.__INFO, 'Saving history')
		self.SaveHistory(history)

		print (self.__INFO, 'Plotting history')
		self.PlotHistory(history)
		
		return history
#======================================================================================
class ContrastiveAETrainer(TrainerValAE):
	def __init__(self, model, epochs:int=150, batch_size:int=100, 
		train_dir:str='./', maxlr:float=1e-3, sfilename:str=None, 
		model_name:str='contrastive_model'):

		super().__init__(model, epochs=epochs, 
			batch_size=batch_size, train_dir=train_dir, 
			maxlr=maxlr, sfilename=sfilename, model_name=model_name)
		self.autoencoder = None
	
	@property
	def Autoencoder(self):
		return self.autoencoder

	@Autoencoder.setter
	def Autoencoder(self, autoencoder):
		self.autoencoder = autoencoder

	def PlotModel(self):
		super().PlotModel()
		if self.Autoencoder is not None and not check_for_modules({'pydot'}):
			tf.keras.utils.plot_model(self.Autoencoder, to_file=os.path.join(self.TrainDir, 'autoencoder.png'), show_shapes=True)

	def SaveModel(self):
		super().SaveModel()
		if self.Autoencoder is not None:
			with open(os.path.join(self.TrainDir, 'AE_summary.txt'), 'w') as summary:
				self.Autoencoder.summary(print_fn=lambda x: summary.write(x + '\n'))
			model_json = self.Autoencoder.to_json()
			with open(os.path.join(self.train_dir, 'autoencoder.json'), "w") as json_file:
				json_file.write(model_json)

			self.Autoencoder.save(os.path.join(self.train_dir, 'autoencoder.h5'))
#======================================================================================
class CategoricalAETrainer(TrainerValAE):
	def __init__(self, model, epochs:int=150, batch_size:int=100, 
		train_dir:str='./', maxlr:float=1e-3, sfilename:str=None, 
		model_name:str='cate_autoencoder'):

		super().__init__(model, epochs=epochs, 
			batch_size=batch_size, train_dir=train_dir, 
			maxlr=maxlr, sfilename=sfilename, model_name=model_name)

		self.categorical_model = None

	@property
	def CategoricalModel(self):
		return self.categorical_model

	@CategoricalModel.setter
	def CategoricalModel(self, categorical_model):
		self.categorical_model = categorical_model

	def PlotModel(self):
		super().PlotModel()
		if self.CategoricalModel is not None and not check_for_modules({'pydot'}):
			tf.keras.utils.plot_model(self.CategoricalModel, to_file=os.path.join(self.TrainDir, 'CatAE.png'), show_shapes=True)

	def SaveModel(self):
		super().SaveModel()
		if self.CategoricalModel is not None:
			with open(os.path.join(self.TrainDir, 'CatAE_summary.txt'), 'w') as summary:
				self.CategoricalModel.summary(print_fn=lambda x: summary.write(x + '\n'))
			model_json = self.CategoricalModel.to_json()
			with open(os.path.join(self.train_dir, 'CatAE.json'), "w") as json_file:
				json_file.write(model_json)

			self.CategoricalModel.save(os.path.join(self.train_dir, 'CatAE.h5'))

#======================================================================================
class DualAETrainer(TrainerValAE):
	def __init__(self, model, epochs:int=150, batch_size:int=100, 
		train_dir:str='./', maxlr:float=1e-3, sfilename:str=None, 
		model_name:str='auto_dualdecoder'):

		super().__init__(model, epochs=epochs, 
			batch_size=batch_size, train_dir=train_dir, 
			maxlr=maxlr, sfilename=sfilename, model_name=model_name)

		self.decoder_2 = None

	@property
	def Decoder2(self):
		return self.decoder_2

	@Decoder2.setter
	def Decoder2(self, decoder_2):
		self.decoder_2 = decoder_2

	def PlotModel(self):
		super().PlotModel()
		if self.Decoder2 is not None and not check_for_modules({'pydot'}):
			tf.keras.utils.plot_model(self.Decoder2, to_file=os.path.join(self.TrainDir, 'decoder_2.png'), show_shapes=True)

	def SaveModel(self):
		super().SaveModel()
		if self.Decoder2 is not None:
			with open(os.path.join(self.TrainDir, 'DEC2_summary.txt'), 'w') as summary:
				self.Decoder2.summary(print_fn=lambda x: summary.write(x + '\n'))
			model_json = self.Decoder2.to_json()
			with open(os.path.join(self.train_dir, 'decoder_2.json'), "w") as json_file:
				json_file.write(model_json)

			self.Decoder2.save(os.path.join(self.train_dir, 'decoder_2.h5'))

#======================================================================================
class TrainerValNN(AbstractTrainer):
	def __init__(self, model, epochs=150, batch_size=100, train_dir='./', maxlr=1e-3, sfilename=None):
		super().__init__(model, epochs=epochs, batch_size=batch_size, train_dir=train_dir, 
						 maxlr=maxlr, sfilename=sfilename)
	"""
	def PlotHistory(self, history):
		print ('[INFO]: Ploting history')
		matplotlib.use('pdf')
		plt.rc('font', family='serif', serif='Times')
		plt.rc('text', usetex=False)
		plt.rc('xtick', labelsize='medium')
		plt.rc('ytick', labelsize='medium')
		plt.rc('axes', labelsize='medium')


		plt.grid(True)
		fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True, sharey=False)
		fig.set_figheight(6)
		#======================================================
		ax1.set_title('Accuracy and Val accuracy')
		ax1.set_ylabel('Magnitude')
		ax1.set_xlabel('Epochs')

		ax2.set_title('Loss and Val loss')
		ax2.set_ylabel('Magnitude')
		ax2.set_xlabel('Epochs')
		
		x_epochs = range(len(history.history['val_accuracy']))
		ax1.plot(x_epochs, history.history['accuracy'], label='accuracy')
		ax1.plot(x_epochs, history.history['val_accuracy'], label='val_accuracy')
		ax1.legend()
		ax2.plot(x_epochs, history.history['loss'], label='loss')
		ax2.plot(x_epochs, history.history['val_loss'], label='val_loss')
		ax2.legend()
		plt.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(self.TrainDir, 'history_training_all.pdf'.format()))
		plt.close()
		#======================================================
		for key, value in history.history.items():
			plt.grid(True)
			plt.xlabel('epochs')
			plt.ylabel(key)
			plt.plot(range(len(value)), value, label=key)
			plt.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(self.TrainDir, 'history_training_{}.pdf'.format(key)))
			plt.close()
	"""
	def PlotModel(self, path_name):
		if not check_for_modules({'pydot'}):
			tf.keras.utils.plot_model(self.Model, to_file=path_name, show_shapes=True)
		else:
			print ('[WARNING]: Some package required to plot the model are missing, model does not be plotted.')
	   
	def SaveModel(self):
		with open(os.path.join(self.train_dir, 'model_summary.txt'), 'w') as summary:
			self.Model.summary(print_fn=lambda x: summary.write(x + '\n'))

		self.PlotModel(os.path.join(self.TrainDir, 'model_image.png'))

		# Generates JSON from model and saves it
		model_json = self.Model.to_json()
		with open(os.path.join(self.train_dir, 'model_json.json'), "w") as json_file:
			json_file.write(model_json)
	
	def train(self, savemodel=True):
		file_name = os.path.join(self.train_dir, self.sfilename)
		check_file_exists(os.path.dirname(file_name))
		
		if savemodel:
			self.SaveModel()

		lr_manager = OneCycleLR(max_lr=self.MaxLR, end_percentage=0.2, 
								scale_percentage=0.1, maximum_momentum=None, 
								minimum_momentum=None,verbose=True, 
								b_size_sent=self.batch_size, sps=self.OCP_sps)

		# Save model every epoch
		save_model = tf.keras.callbacks.ModelCheckpoint(file_name)
		callbacks=[save_model, lr_manager]
		for callback in self.callbacks_list:
			callbacks.append(callback)

		print ('[INFO]: Using {}'.format(self.__class__.__name__))

		history = self.Model.fit(self.x_train_dict, self.y_train_dict, 
					validation_data=(self.x_val_dict, self.y_val_dict) if self.x_val_dict is not None and self.y_val_dict is not None else None, 
					batch_size=self.batch_size, epochs=self.epochs,
					callbacks=callbacks)
		
		print ('[INFO]: Saving history')
		self.SaveHistory(history)
		print ('[INFO]: Plotting history')
		self.PlotHistory(history)
		
		return history
#======================================================================================
class TrainerValnonOCP(TrainerValNN):
	def __init__(self, model, epochs=150, batch_size=100, train_dir='./', maxlr=1e-3, sfilename=None):
		super().__init__(model, epochs=epochs, batch_size=batch_size, train_dir=train_dir, 
						 maxlr=maxlr, sfilename=sfilename)
						 
	def train(self, savemodel=True):
		file_name = os.path.join(self.train_dir, self.sfilename)
		check_file_exists(os.path.dirname(file_name))
		
		if savemodel:
			self.SaveModel()

		# Save model every epoch
		save_model = tf.keras.callbacks.ModelCheckpoint(file_name)
		callbacks=[save_model]
		for callback in self.callbacks_list:
			callbacks.append(callback)

		print ('[INFO]: Using {}'.format(self.__class__.__name__))

		history = self.Model.fit(self.x_train_dict, self.y_train_dict, 
					validation_data=(self.x_val_dict, self.y_val_dict) if self.x_val_dict is not None and self.y_val_dict is not None else None, 
					batch_size=self.batch_size, epochs=self.epochs,
					callbacks=callbacks)
		
		print ('[INFO]: Saving history')
		self.SaveHistory(history)

		print ('[INFO]: Plotting history')
		self.PlotHistory(history)
		
		return history
#======================================================================================
class GETrainerValNN(TrainerValNN):
	def __init__(self, model, epochs=150, batch_size=100, train_dir='./', maxlr=1e-3, 
				 sfilename=None):
		super().__init__(model, epochs=epochs, batch_size=batch_size, train_dir=train_dir, 
						 maxlr=maxlr, sfilename=sfilename)
		
	"""
	def PlotHistory(self, history):
		print ('[INFO]: Ploting history')
		matplotlib.use('pdf')
		plt.rc('font', family='serif', serif='Times')
		plt.rc('text', usetex=False)
		plt.rc('xtick', labelsize='medium')
		plt.rc('ytick', labelsize='medium')
		plt.rc('axes', labelsize='medium')


		plt.grid(True)
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1,sharex=True, sharey=False)
		fig.set_figheight(6)
		#======================================================
		ax1.set_title('Accuracy and Val accuracy')
		ax1.set_ylabel('Magnitude')
		ax1.set_xlabel('Epochs')

		ax2.set_title('Loss and Val loss')
		ax2.set_ylabel('Magnitude')
		ax2.set_xlabel('Epochs')
		
		x_epochs = range(len(history.history['val_accuracy']))
		ax1.plot(x_epochs, history.history['accuracy'], label='accuracy')
		ax1.plot(x_epochs, history.history['val_accuracy'], label='val_accuracy')
		ax1.legend()
		ax2.plot(x_epochs, history.history['loss'], label='loss')
		ax2.plot(x_epochs, history.history['val_loss'], label='val_loss')
		ax2.legend()
		ax3.plot(x_epochs, history.history['key_rank'], label='key_rank')
		ax3.plot(x_epochs, history.history['val_key_rank'], label='val_key_rank')
		ax3.legend()

		plt.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(self.TrainDir, 'history_training_all.pdf'.format()))
		plt.close()
		#======================================================
		for key, value in history.history.items():
			plt.grid(True)
			plt.xlabel('epochs')
			plt.ylabel(key)
			plt.plot(range(len(value)), value, label=key)
			plt.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(self.TrainDir, 'history_training_{}.pdf'.format(key)))
			plt.close()
		#======================================================
	"""
	def SaveModel(self):
		with open(os.path.join(self.train_dir, 'model_summary.txt'), 'w') as summary:
			self.Model.summary(print_fn=lambda x: summary.write(x + '\n'))

		self.PlotModel(os.path.join(self.TrainDir, 'model_image.png'))

		# Generates JSON from model and saves it
		model_json = self.Model.to_json()
		with open(os.path.join(self.train_dir, 'model_json.json'), "w") as json_file:
			json_file.write(model_json)
	
	def train(self, savemodel=True):
		file_name = os.path.join(self.train_dir, self.sfilename)
		check_file_exists(os.path.dirname(file_name))
		
		if savemodel:
			self.SaveModel()

		lr_manager = OneCycleLR(max_lr=self.MaxLR, end_percentage=0.2, 
								scale_percentage=0.1, maximum_momentum=None, 
								minimum_momentum=None,verbose=True, 
								b_size_sent=self.batch_size, sps=self.OCP_sps)

		# Save model every epoch
		save_model = tf.keras.callbacks.ModelCheckpoint(file_name)
		callbacks=[save_model, lr_manager]
		for callback in self.callbacks_list:
			callbacks.append(callback)

		print ('[INFO]: Using {}'.format(self.__class__.__name__))

		history = self.Model.fit(self.x_train_dict, self.y_train_dict, 
					validation_data=(self.x_val_dict, self.y_val_dict) if self.x_val_dict is not None and self.y_val_dict is not None else None, 
					batch_size=self.batch_size, epochs=self.epochs,
					callbacks=callbacks)
		
		print ('[INFO]: Saving history')
		self.SaveHistory(history)

		print ('[INFO]: Plotting history')
		self.PlotHistory(history)

		return history
#======================================================================================