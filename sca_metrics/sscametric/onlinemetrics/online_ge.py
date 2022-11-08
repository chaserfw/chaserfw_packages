from tensorflow.python.keras import backend as K
import tensorflow.keras as tk
import tensorflow as tf
import numpy as np
from .. import guessing_entropy


# ================= Core =====================
# calculate key prob for all keys
def calculate_key_prob(y_true, y_pred, nb_traces_attacks, nb_attacks, correct_key, 
                       attack_byte=2, classes=256, leakage='ID', shuffle=True, 
                       output_rank=False):
    plt_attack = y_true[:, classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        if y_pred.shape[1] > classes: 
            y_pred = y_pred[:, :classes]
        GE = guessing_entropy.perform_attacks(nb_traces_attacks, y_pred, plt_attack[:, 1:],
                                              correct_key=correct_key,
                                              leakage=leakage,
                                              nb_attacks=nb_attacks, 
                                              byte=attack_byte,
                                              shuffle=shuffle,
                                              output_rank=output_rank)
    else:  # otherwise, return zeros
        GE = np.float32(np.zeros(256))
    return GE


@tf.function
def tf_calculate_key_prob(y_true, y_pred, nb_traces_attacks, nb_attacks, correct_key, attack_byte=2, 
                          classes=256, leakage='ID', shuffle=True, output_rank=False):
    _ret = tf.numpy_function(calculate_key_prob, 
                            [y_true, y_pred, nb_traces_attacks, nb_attacks, 
                            correct_key, attack_byte, classes, leakage, 
                            shuffle, output_rank], 
                            tf.float32)
    return _ret

class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, correct_key, nb_traces_attacks, nb_attacks, attack_byte=2, 
                    classes=256, name='key_rank', output_rank=False, **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.correct_key       = correct_key
        self.nb_traces_attacks = nb_traces_attacks
        self.nb_attacks        = nb_attacks
        self.attack_byte       = attack_byte
        self.classes           = classes
        self.output_rank       = output_rank

        self.acc_sum = self.add_weight(name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred, self.nb_traces_attacks, 
                                                      self.nb_attacks, self.correct_key, 
                                                      self.attack_byte, self.classes,
                                                      output_rank=self.output_rank))

    def result(self):
        # GE as objective
        return tf.numpy_function(guessing_entropy.rk_key, [self.acc_sum, self.correct_key], tf.float32)
        # Lm as objective
        # return tf.numpy_function(calculate_Lm, [self.acc_sum], tf.float32)

    def reset_state(self):
        self.acc_sum.assign(K.zeros(256))