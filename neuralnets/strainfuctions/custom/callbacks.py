import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sscametric import perform_attacks

class EpochCounterCallback(tf.keras.callbacks.Callback):
	def __init__(self, init_value=0):
		super().__init__()
		self.counter = K.variable(value=init_value)
		
	def get_counter(self):
		return self.counter
	
	def on_epoch_end(self, epoch, logs={}):
		K.set_value(self.counter, self.counter+1)

#===============================================================================
class KeyRankingCallbackEpochEnd(tf.keras.callbacks.Callback):
	def __init__(self, truncated_model, attack_traces, plt_attack, 
				 nb_traces_attacks, correct_key, nb_attacks, attack_byte=2,
				 output_rank=True, 
				 persistence=None,
				 patient=None,
				 es=False,
				 GE_tolerant=None,
				 minimal_trace=None,
				 ):
		self.truncated_model   = truncated_model
		self.current_epoch     = 0
		self.plt_attack        = plt_attack
		self.attack_traces     = attack_traces
		self.key_ranking       = []
		self.correct_key       = correct_key
		self.nb_attacks        = nb_attacks
		self.attack_byte       = attack_byte
		self.output_rank       = output_rank
		self.nb_traces_attacks = nb_traces_attacks
		
		self.es              = es
		self.persistence     = persistence if persistence is not None else self.nb_traces_attacks
		self.patient         = patient if patient is not None else 3
		self.patient_counter = 0
		self.GE_tolerant     = GE_tolerant if GE_tolerant is not None else 0
		self.minimal_trace   = minimal_trace
		self.OK_criterion    = False

	def on_epoch_end(self, epoch, logs={}):
		predictions = self.truncated_model.predict(self.attack_traces)
		predictions = predictions[:,:256]
		avg_rank = np.array(perform_attacks(self.nb_traces_attacks, 
											predictions, 
											self.plt_attack,
											correct_key=self.correct_key,
											nb_attacks=self.nb_attacks, 
											byte=self.attack_byte,
											output_rank=self.output_rank))

		
		tf.print ('mean',np.mean(avg_rank))
		tf.print (avg_rank)
		self.key_ranking.append(avg_rank)

		if self.es:
			if self.minimal_trace is None: # CASE 1: Naive implementation - minimal trace is not specified, 
										   # the lowerbound would be the position where GE_tolerant is located
				GE = tf.convert_to_tensor(avg_rank, dtype=tf.float32)# Remove when developing GE function base on 
																	 # tensors
				index = tf.cast(tf.where(GE <= self.GE_tolerant), dtype=tf.int32)
				if (index.shape[0] > 0):
					goal_counter = tf.cast(tf.reduce_all(tf.slice(GE, 
												index[0], self.persistence-index[0]) <= self.GE_tolerant), 
												tf.int8)
					if goal_counter > 0: # Keeping patient
						self.patient_counter += goal_counter
					else: # Restart counter, patient would not have been reached
						self.patient_counter = 0

					if self.patient_counter == self.patient: # OK Criterion
						print("Epoch %05d: early stopping threshold" % epoch)
						self.model.stop_training = True
				



	def get_key_ranking(self):
		return self.key_ranking
#===============================================================================