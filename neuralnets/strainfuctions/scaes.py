import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from keras.utils.generic_utils import has_arg
import numpy as np
import datetime
import logging
import random
import math
import sys
import os

#----------------------------------------------------------------------
# Core
#----------------------------------------------------------------------

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    ], dtype=np.uint8)
#-------------------------------------------------------------------------------
def compute_epsilon(predictions, trace):
    min_proba_predictions = predictions[trace][np.array(predictions[trace]) != 0]
    if len(min_proba_predictions) == 0:
        print("Error: got a prediction with only zeroes ... this should not happen!")
        sys.exit(-1)
    min_proba = min(min_proba_predictions)
    return np.log(min_proba**2)
#-------------------------------------------------------------------------------
# Compute the rank of the real key for a give set of predictions
def rank_cubic(predictions, plt_attack, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, attack_byte):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba
    
    plaintexts = plt_attack[min_trace_idx:max_trace_idx][:,attack_byte]
    
    for i, plaintext in enumerate(plaintexts):
        # Our candidate key byte probability is the sum of the predictions logs
        proba = predictions[i][AES_Sbox[np.bitwise_xor(plaintext, compute_ge.key_candidates)]]
        value = np.where(proba != 0, np.log(proba), compute_epsilon(predictions, i))
        key_bytes_proba = np.add(key_bytes_proba, value)
        
    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)
#-------------------------------------------------------------------------------
def compute_ge(predictions, plt_attack, attack_byte, real_key, min_trace_idx, max_trace_idx, rank_step):
    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    # Computing the 256 vector for indexing
    compute_ge.key_candidates = range(0, 256)
    
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank_cubic(predictions[t-rank_step:t], plt_attack, real_key, t-rank_step, t, key_bytes_proba, attack_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks

#----------------------------------------------------------------------
# SCAESPolicy
#----------------------------------------------------------------------
class SCAESPolicy(tf.keras.callbacks.Callback):
    def __init__(self, attack_traces, plt_attack, 
                 nb_traces_attacks, correct_key, 
                 nb_attacks, attack_byte=2,
                 persistence=None, patience=3,
                 es=False, w=0,
                 minimal_trace=None, 
                 binary_persistence=0.95,
                 rank_steps=10, 
                 external_model=None):

        self.current_epoch      = 0
        self.plt_attack         = plt_attack
        self.attack_traces      = attack_traces
        self.ge_log             = []
        self.correct_key        = correct_key
        self.nb_attacks         = nb_attacks
        self.attack_byte        = attack_byte
        self.nb_traces_attacks  = nb_traces_attacks

        self.es                 = es
        self.persistence        = math.floor(persistence if persistence is not None else self.nb_traces_attacks / rank_steps) - 1
        self.patience           = patience
        self.patience_counter   = 0
        self.w                  = w
        self.minimal_trace      = minimal_trace
        self.OK_criterion       = False
        self.binary_persistence = binary_persistence
        self.this_model         = external_model if external_model is not None else self.model
        self.x_ticks = None

    def on_epoch_end(self, epoch, logs={}):
        avg_cumulative = None
        x_ticks        = None
        for i in range(self.nb_attacks):
            trace_index = random.sample(range(self.attack_traces.shape[0]), self.nb_traces_attacks)
            true_attack_traces = self.attack_traces[trace_index]
            chosen_plt_attack = self.plt_attack[trace_index]

            predictions = self.this_model.predict(true_attack_traces)
            avg_rank = np.array(compute_ge(predictions,
                                                plt_attack=chosen_plt_attack,
                                                attack_byte=self.attack_byte,
                                                real_key=self.correct_key,
                                                min_trace_idx = 0, 
                                                max_trace_idx = self.nb_traces_attacks, 
                                                rank_step = 10))

            if avg_cumulative is None:
                x_ticks = avg_rank[:,0]
                avg_cumulative = np.zeros(shape=(len(x_ticks)), dtype=np.uint32)
            
            avg_cumulative = np.add(avg_cumulative, avg_rank[:,1])
        
        avg_rank = np.true_divide(avg_cumulative, self.nb_attacks, dtype=np.float32)
        tf.print ('[INFO]: Mean', np.mean(avg_rank))
        self.ge_log.append(np.array([avg_rank, self.x_ticks]))

        if self.es:
            GE = tf.convert_to_tensor(avg_rank, dtype=tf.float32)# Remove when developing GE function base on tensors
            
            # CASE 1: soft - minimal trace is not specified, 
            # the lowerbound would be the position where GE_tolerant is located
            # CASE 2: greedy - if index is below or same as the minimal trace then 
            # index is taken as the starting trace to slice the GE, if index is above 
            # then an invalid index that counted as zero hit is returned.
            convergence_index = tf.where(GE <= self.w)
            index = tf.cond(convergence_index.shape[0] > 0 and self.minimal_trace is None, lambda: convergence_index[0], 
                    lambda: tf.cond(convergence_index.shape[0] > 0 and convergence_index[0] <= self.minimal_trace, 
                                    lambda: convergence_index[0],  lambda: tf.constant([-1])))

            after_first_zero_slice = lambda index: tf.slice(GE, index, self.persistence-index) <= self.w
            hit = tf.cond(index >= 0, lambda: tf.cond(self.binary_persistence is None, 
                                                      lambda: tf.cast(tf.reduce_all(after_first_zero_slice(index), 
                                                                                    tf.float32)),
                                                      lambda: tf.cast(tf.reduce_mean(tf.cast(after_first_zero_slice(index), 
                                                                                             tf.float32)) >= self.binary_persistence, 
                                                                      tf.float32)), lambda: 0)
            self.patience_counter = (self.patience_counter + hit) * hit
            if self.patience_counter >= self.patience: # OK Criterion
                tf.print("Epoch %03d: early stopping threshold" % epoch)
                self.model.stop_training = True
    
    def reset(self):
        self.ge_log          = []
        self.patience_counter = 0
        self.OK_criterion    = False    
    
    def get_ge_log(self):
        return self.ge_log

#----------------------------------------------------------------------
# Customized Grid Search compatible with SCA-ES
#----------------------------------------------------------------------
class SCAESGridSearch():
    def __init__(self, keras_estimator, save_ge_logs=False, save_iter_log=False):
        '''
        '''
        
        self.keras_estimator = keras_estimator
        self.__params_set    = keras_estimator.get_params()
        self.__build_fn      = self.__params_set['build_fn']
        self.save_ge_logs    = save_ge_logs
        self.save_iter_log   = save_iter_log

        del self.__params_set['build_fn']
        self.__dest_path_auto_checkpoint_cbk = None
        self.__auto_chechpoint_cbk_params    = None
        self.__sca_es_policy                 = None
    
    def set_auto_checkpoint_callback(self, dest_path, **callback_params):
        self.__dest_path_auto_checkpoint_cbk = os.path.join(dest_path, 'model_{}_it_{}.h5')
        self.__auto_chechpoint_cbk_params    = callback_params
        
    def __filter_params(self, fn, params):
        res = {}
        for name, value in params.items():
            if has_arg(fn, name):
                res.update({name: value})
        return res
    
    def __date_formatter(self):
        """Returns the data and time in a safe format to use as part of the a file name
        """
        currentDT = datetime.datetime.now()
        return currentDT.strftime("%Y-%m-%d %H:%M:%S").replace('-', '').replace(':', '').replace(' ', '-')
        
    def fit(self, **kwargs):
        # Initializate a log session
        if self.save_iter_log:
            logging.basicConfig(filename=os.path.join(os.path.abspath(os.path.dirname(self.__dest_path_auto_checkpoint_cbk)), 
                                                      'sca-es_grid_search_{}.log'.format(self.__date_formatter())), 
                                filemode='w', 
                                level=logging.DEBUG)
        
        callback_filtered = []
        if 'callbacks' in kwargs and self.__dest_path_auto_checkpoint_cbk is not None:
            for callback in kwargs['callbacks']:
                # Remove ModelCheckpoint from the user callback original parameter to replace it for the auto_checkpoint
                if not isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                    callback_filtered.append(callback)
                if isinstance(callback, SCAESPolicy):
                    self.__sca_es_policy = callback

        param_list = ParameterGrid(self.__params_set)
        for i, l in enumerate(param_list):
            model = self.__build_fn(**self.__filter_params(self.__build_fn, l))            
            if self.__dest_path_auto_checkpoint_cbk is not None:
                final_callbacks_list = [tf.keras.callbacks.ModelCheckpoint(self.__dest_path_auto_checkpoint_cbk.format(model.name, i), 
                                                                          **self.__auto_chechpoint_cbk_params)] + callback_filtered
                kwargs.update({'callbacks': final_callbacks_list})
            
            condensed_parameters = dict(kwargs, **l)
            # Fit the model with the actual parameters combination
            print ('[INFO]: Training model:', model.name, 'iter:', str(i))
            model.fit(**self.__filter_params(model.fit, condensed_parameters))
            
            if self.save_iter_log:
                logging.info('it-{} : {}'.format(str(i), condensed_parameters))
            
            if self.__sca_es_policy is not None:
                if self.save_ge_logs:
                    print ('[INFO]: saving SCA-ES policy guessing entropy logs')
                    np.save('{}_it_{}_ge_log'.format(os.path.join(os.path.abspath(os.path.dirname(self.__dest_path_auto_checkpoint_cbk)), 
                                                                  model.name), 
                                                     str(i)),
                            self.__sca_es_policy.get_ge_log())
                self.__sca_es_policy.reset()
                
            # Check if model early stopped
            if model.stop_training:
                print ('[INFO]: Model early stopped')
                logging.info('[INFO]: Model early stopped')
                break
