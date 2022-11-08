import matplotlib
import matplotlib.pyplot as plt
import abc
import pickle
import os
class AbstractTrainer(metaclass=abc.ABCMeta):
	def __init__(self, model, epochs=150, 
		batch_size=100, train_dir='./', 
		maxlr=1e-3, sfilename=None, 
		**kwargs):

		self.model          = model
		self.epochs         = epochs
		self.batch_size     = batch_size
		self.train_dir      = train_dir
		self.maxlr          = maxlr
		self.OCP_sps        = None
		self.sfilename      = sfilename
		self.callbacks_list = []

		self.x_train_dict = None
		self.y_train_dict = None
		self.x_val_dict   = None
		self.y_val_dict   = None

		self._height_factor = 3
		if 'height_factor' in kwargs:
			self._height_factor = kwargs['height_factor']
		
	
	@abc.abstractmethod
	def train(self, savemodel=True):
		pass
	
	@property
	def Epochs(self):
		return self.epochs
	
	@Epochs.setter
	def Epochs(self, epochs):
		self.epochs = epochs
		
	@property
	def BatchSize(self):
		return self.batch_size
	
	@BatchSize.setter
	def BatchSize(self, batch_size):
		self.batch_size = batch_size
		
	@property
	def TrainDir(self):
		return self.train_dir
	
	@TrainDir.setter
	def TrainDir(self, train_dir):
		self.train_dir = train_dir
	
	@property
	def MaxLR(self):
		return self.maxlr
	
	@MaxLR.setter
	def MaxLR(self, maxlr):
		self.epochs = maxlr
		
	@property
	def Model(self):
		return self.model
	
	@property
	def XTrainDict(self):
		return self.x_train_dict
	
	@XTrainDict.setter
	def XTrainDict(self, x_train_dict):
		self.x_train_dict = x_train_dict
	
	@property
	def YTrainDict(self):
		return self.y_train_dict
	
	@YTrainDict.setter
	def YTrainDict(self, y_train_dict):
		self.y_train_dict = y_train_dict
	
	@property
	def XValDict(self):
		return self.x_val_dict
	
	@XValDict.setter
	def XValDict(self, x_val_dict):
		self.x_val_dict = x_val_dict
	
	@property
	def YValDict(self):
		return self.y_val_dict
	
	@YValDict.setter
	def YValDict(self, y_val_dict):
		self.y_val_dict = y_val_dict
		
	@property
	def OCPsps(self):
		return self.OCP_sps
		
	@OCPsps.setter
	def OCPsps(self, OCP_sps):
		self.OCP_sps = OCP_sps        
		
	def addCallback(self, callback):
		self.callbacks_list.append(callback)
		
	def SaveModel(self, file_name):
		pass
	
	def SaveHistory(self, history):
		hitory_file = os.path.join(self.TrainDir, 'history')
		with open(hitory_file, 'wb') as file_pi:
			pickle.dump(history.history, file_pi)

	def PlotHistory(self, history):
		matplotlib.use('pdf')
		plt.rc('font', family='serif', serif='Times')
		plt.rc('text', usetex=False)
		plt.rc('xtick', labelsize='medium')
		plt.rc('ytick', labelsize='medium')
		plt.rc('axes', labelsize='medium')

		fig, ax = plt.subplots(len(self.Model.metrics), 1, sharex=True, sharey=False)
		fig.set_figheight(len(self.Model.metrics)*self._height_factor)
		ax = list(ax) if len(self.Model.metrics) > 1 else [ax]
		for index, metric in enumerate(self.Model.metrics):
			index_name = metric.name
			x_epochs = range(len(history.history[index_name]))
			ax[index].plot(x_epochs, history.history[index_name], label=index_name)
			if self.XValDict is not None and self.YValDict is not None:
				ax[index].plot(x_epochs, history.history['{}_{}'.format('val', index_name)], label='{}_{}'.format('val', index_name))
				ax[index].set_title('{} and val {}'.format(index_name, index_name))
			else:
				ax[index].set_title('{}'.format(index_name))

			ax[index].set_ylabel('Magnitude')
			ax[index].set_xlabel('Epochs')
			ax[index].legend()
			ax[index].grid(True)

		plt.grid(True)
		plt.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(self.TrainDir, 'history_all_metrics.pdf'.format()))
		plt.close()
		#======================================================
		for key, value in history.history.items():
			plt.grid(True)
			plt.xlabel('Epochs')
			plt.ylabel(key)
			plt.plot(range(len(value)), value, label=key)
			plt.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(self.TrainDir, 'history_{}.pdf'.format(key)))
			plt.close()
