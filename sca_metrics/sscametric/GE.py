from sutils import trange
import numpy as np
import random
import sys


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
def _compute_epsilon(predictions, trace):
    min_proba_predictions = predictions[trace][np.array(predictions[trace]) != 0]
    if len(min_proba_predictions) == 0:
        print("Error: got a prediction with only zeroes ... this should not happen!")
        sys.exit(-1)
    min_proba = min(min_proba_predictions)
    return np.log(min_proba**2)
#-------------------------------------------------------------------------------
# Compute the rank of the real key for a give set of predictions
def _rank_cubic(predictions, plt_attack, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, attack_byte):
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
        proba = predictions[i][AES_Sbox[np.bitwise_xor(plaintext, _compute_ge.key_candidates)]]
        value = np.where(proba != 0, np.log(proba), _compute_epsilon(predictions, i))
        key_bytes_proba = np.add(key_bytes_proba, value)
        
    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)
#-------------------------------------------------------------------------------
def _compute_ge(predictions, plt_attack, attack_byte, real_key, min_trace_idx, 
                max_trace_idx, rank_step):
    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    # Computing the 256 vector for indexing
    _compute_ge.key_candidates = range(0, 256)
    
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = _rank_cubic(predictions[t-rank_step:t], plt_attack, 
                                                     real_key, t-rank_step, t, key_bytes_proba, 
                                                     attack_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks

def compute(classifier, nb_attacks, rank_steps, nb_traces_attacks, 
            attack_traces, plt_attack, attack_byte, correct_key, 
            pbar=False):
    """Computes the guessing entropy using a previously trained classifier

        :param classifier: A keras model
        :type classifier: tf.keras.model.Model

        :param nb_attacks: Number of attacks to average the result
        :type nb_attacks: int

        :param rank_steps: the size of the attack mini batch
        :type rank_steps: int

        :param nb_traces_attacks: Number of traces to compute the GE
        :type nb_traces_attacks: int

        :param attack_traces: Set of traces used to compute the prediction vector
        :type attack_traces: numpy.ndarray

        :param plt_attack: Set of plaintext match-indexed with the set of traces
        :type plt_attack: numpy.ndarray

        :param correct_key: The byte of the correct key.
        :type correct_key: int

        :param pbar: To display or not a progress bar (it is recommended to set False 
                     when using as a early stopping).
        :type pbar: bool

        :return: A numpy array comprises the averaged guessing entropy and the ticks 
                 in number of traces.
        :rtype: numpy.ndarray
    """
    itrange = trange(nb_attacks, desc="[INFO]: Computing guessing entropy") if pbar else \
        range(nb_attacks)
    x_ticks = None
    for i in itrange:
        trace_index = random.sample(range(attack_traces.shape[0]), nb_traces_attacks)
        true_attack_traces = attack_traces[trace_index]
        chosen_plt_attack = plt_attack[trace_index]

        predictions = classifier.predict(true_attack_traces)
        avg_rank = np.array(_compute_ge(predictions,
                                        plt_attack=chosen_plt_attack,
                                        attack_byte=attack_byte,
                                        real_key=correct_key,
                                        min_trace_idx=0, 
                                        max_trace_idx=nb_traces_attacks, 
                                        rank_step=rank_steps))
        
        if x_ticks is None:
            x_ticks = avg_rank[:,0]
            avg_cumulative = np.zeros(shape=(len(x_ticks)), dtype=np.uint32)
        avg_cumulative = np.add(avg_cumulative, avg_rank[:,1])
    
    avg_rank = np.true_divide(avg_cumulative, nb_attacks, dtype=np.float32)
    return np.array([avg_rank, x_ticks])