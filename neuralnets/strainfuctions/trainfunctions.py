import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import pickle
# Code implemented by https://github.com/titu1994/keras-one-cycle
# Code is ported from https://github.com/fastai/fastai
class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self,
                 max_lr,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True,
                 b_size_sent=None,
                 sps=None):
        """ This callback implements a cyclical learning rate policy (CLR).
        This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
        After the completion of 1 cycle, the learning rate will decrease rapidly to
        100th its initial lowest value.

        # Arguments:
            max_lr: Float. Initial learning rate. This also sets the
                starting learning rate (which will be 10x smaller than
                this), and will increase to this value during the first cycle.
            end_percentage: Float. The percentage of all the epochs of training
                that will be dedicated to sharply decreasing the learning
                rate after the completion of 1 cycle. Must be between 0 and 1.
            scale_percentage: Float or None. If float, must be between 0 and 1.
                If None, it will compute the scale_percentage automatically
                based on the `end_percentage`.
            maximum_momentum: Optional. Sets the maximum momentum (initial)
                value, which gradually drops to its lowest value in half-cycle,
                then gradually increases again to stay constant at this max value.
                Can only be used with SGD Optimizer.
            minimum_momentum: Optional. Sets the minimum momentum at the end of
                the half-cycle. Can only be used with SGD Optimizer.
            verbose: Bool. Whether to print the current learning rate after every
                epoch.

        # Reference
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        """
        super(OneCycleLR, self).__init__()

        if end_percentage < 0. or end_percentage > 1.:
            raise ValueError("`end_percentage` must be between 0 and 1")

        if scale_percentage is not None and (scale_percentage < 0. or scale_percentage > 1.):
            raise ValueError("`scale_percentage` must be between 0 and 1")

        self.initial_lr = max_lr
        self.end_percentage = end_percentage
        self.scale = float(scale_percentage) if scale_percentage is not None else float(end_percentage)
        self.max_momentum = maximum_momentum
        self.min_momentum = minimum_momentum
        self.verbose = verbose

        if self.max_momentum is not None and self.min_momentum is not None:
            self._update_momentum = True
        else:
            self._update_momentum = False

        self.clr_iterations = 0.
        self.history = {}

        self.epochs = None
        self.batch_size = None
        self.samples = None
        self.steps = None
        self.num_iterations = None
        self.mid_cycle_id = None
        self.b_size_sent = b_size_sent
        self.sps = sps

    def _reset(self):
        """
        Reset the callback.
        """
        self.clr_iterations = 0.
        self.history = {}

    def compute_lr(self):
        """
        Compute the learning rate based on which phase of the cycle it is in.

        - If in the first half of training, the learning rate gradually increases.
        - If in the second half of training, the learning rate gradually decreases.
        - If in the final `end_percentage` portion of training, the learning rate
            is quickly reduced to near 100th of the original min learning rate.

        # Returns:
            the new learning rate
        """
        if self.clr_iterations > 2 * self.mid_cycle_id:
            current_percentage = (self.clr_iterations - 2 * self.mid_cycle_id)
            current_percentage /= float((self.num_iterations - 2 * self.mid_cycle_id))
            new_lr = self.initial_lr * (1. + (current_percentage *
                                              (1. - 100.) / 100.)) * self.scale

        elif self.clr_iterations > self.mid_cycle_id:
            current_percentage = 1. - (
                self.clr_iterations - self.mid_cycle_id) / self.mid_cycle_id
            new_lr = self.initial_lr * (1. + current_percentage *
                                        (self.scale * 100 - 1.)) * self.scale

        else:
            current_percentage = self.clr_iterations / self.mid_cycle_id
            new_lr = self.initial_lr * (1. + current_percentage *
                                        (self.scale * 100 - 1.)) * self.scale

        if self.clr_iterations == self.num_iterations:
            self.clr_iterations = 0

        return new_lr

    def compute_momentum(self):
        """
         Compute the momentum based on which phase of the cycle it is in.

        - If in the first half of training, the momentum gradually decreases.
        - If in the second half of training, the momentum gradually increases.
        - If in the final `end_percentage` portion of training, the momentum value
            is kept constant at the maximum initial value.

        # Returns:
            the new momentum value
        """
        if self.clr_iterations > 2 * self.mid_cycle_id:
            new_momentum = self.max_momentum

        elif self.clr_iterations > self.mid_cycle_id:
            current_percentage = 1. - ((self.clr_iterations - self.mid_cycle_id) / float(
                                        self.mid_cycle_id))
            new_momentum = self.max_momentum - current_percentage * (
                self.max_momentum - self.min_momentum)

        else:
            current_percentage = self.clr_iterations / float(self.mid_cycle_id)
            new_momentum = self.max_momentum - current_percentage * (
                self.max_momentum - self.min_momentum)

        return new_momentum

    def on_train_begin(self, logs={}):
        logs = logs or {}
        print ('Callback parameter:', self.params)
        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size'] if 'batch_size' in self.params.keys() else self.b_size_sent
        self.samples = self.params['samples'] if 'samples' in self.params.keys() else self.sps
        self.steps = self.params['steps']

        if self.steps is not None:
            self.num_iterations = self.epochs * self.steps
        else:
            if (self.samples % self.batch_size) == 0:
                remainder = 0
            else:
                remainder = 1
            self.num_iterations = (self.epochs + remainder) * self.samples // self.batch_size

        self.mid_cycle_id = int(self.num_iterations * ((1. - self.end_percentage)) / float(2))

        self._reset()
        tf.keras.backend.set_value(self.model.optimizer.lr, self.compute_lr())

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()
            tf.keras.backend.set_value(self.model.optimizer.momentum, new_momentum)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.clr_iterations += 1
        new_lr = self.compute_lr()

        self.history.setdefault('lr', []).append(
            tf.keras.backend.get_value(self.model.optimizer.lr))
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()

            self.history.setdefault('momentum', []).append(
                tf.keras.backend.get_value(self.model.optimizer.momentum))
            tf.keras.backend.set_value(self.model.optimizer.momentum, new_momentum)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self._update_momentum:
                print(" - lr: %0.5f - momentum: %0.2f " %
                      (self.history['lr'][-1], self.history['momentum'][-1]))

            else:
                print(" - lr: %0.5f " % (self.history['lr'][-1]))


################################################################################
def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return
################################################################################
def train_model_with_val(X_profiling, X2_profiling, Y_profiling, X_validation, X2_validation, Y_validation, model, save_file_name, epochs=150, batch_size=100, train_dir='./'):
    """Funcion que entrena el modelo espera un conjunto de validación
    """
    save_file_name = os.path.join(train_dir, save_file_name)
    check_file_exists(os.path.dirname(save_file_name))
    # Save model every epoch
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
    callbacks=[save_model]
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[0][1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[0][1], len(X_profiling[0])))
        sys.exit(-1)
        
    print ('[INFO]: Using train_model_with_val')
    Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    Reshaped_X2_profiling = X2_profiling.reshape((X2_profiling.shape[0], X2_profiling.shape[1], 1))
    Reshaped_X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
    Reshaped_X2_validation = X2_validation.reshape((X2_validation.shape[0], X2_validation.shape[1], 1))
    
    history = model.fit({'input_1': Reshaped_X_profiling, 'input_2': Reshaped_X2_profiling}, 
                        {'predictions': tf.keras.utils.to_categorical(Y_profiling, num_classes=256)}, 
                        validation_data=({'input_1':Reshaped_X_validation, 'input_2': Reshaped_X2_validation}, 
                                         {'predictions': tf.keras.utils.to_categorical(Y_validation, num_classes=256)}), 
                        batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    
    return history

################################################################################
def train_model(X_profiling, X2_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, train_dir='./'):
    save_file_name = os.path.join(train_dir, save_file_name)
    check_file_exists(os.path.dirname(save_file_name))
    
    # Save model every epoch
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
    callbacks=[save_model]
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[0][1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[0][1], len(X_profiling[0])))
        sys.exit(-1)
        
    print ('[INFO]: Using train_model')
    Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    Reshaped_X2_profiling = X2_profiling.reshape((X2_profiling.shape[0], X2_profiling.shape[1], 1))
        
    history = model.fit({'input_1': Reshaped_X_profiling, 'input_2': Reshaped_X2_profiling}, 
                        {'predictions': tf.keras.utils.to_categorical(Y_profiling, num_classes=256)},
                        batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return history


################################################################################
def train_model_dict_with_val(x_train_dict, y_train_dict, x_val_dict, y_val_dict, model, save_file_name, epochs=150, batch_size=100, train_dir='./'):
    save_file_name = os.path.join(train_dir, save_file_name)
    check_file_exists(os.path.dirname(save_file_name))
        
    # Save model every epoch
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
    callbacks=[save_model]
        
    print ('[INFO]: Using train_model_dict_with_val')
    history = model.fit(x_train_dict, y_train_dict, validation_data=(x_val_dict, y_val_dict), 
                        batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return history

################################################################################
def train_model_dict_with_val_maxlr(x_train_dict, y_train_dict, x_val_dict, y_val_dict, model, save_file_name, epochs=150, b_size=100, train_dir='./', maxlr=1e-3):
    save_file_name = os.path.join(train_dir, save_file_name)
    check_file_exists(os.path.dirname(save_file_name))
    
    lr_manager = OneCycleLR(max_lr=maxlr, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None,verbose=True, 
                            b_size_sent=b_size, sps=x_train_dict['input_1'].shape[0])
            
    # Save model every epoch
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
    callbacks=[save_model, lr_manager]
    
    print ('[INFO]: batch_size:', b_size)
    print ('[INFO]: Using train_model_dict_with_val_maxlr')
    history = model.fit(x_train_dict, y_train_dict, validation_data=(x_val_dict, y_val_dict), 
                        batch_size=b_size, epochs=epochs, callbacks=callbacks)
    return history

################################################################################
def train_model_dict(x_train_dict, y_train_dict, model, save_file_name, epochs=150, batch_size=100, train_dir='./', val_split=0):
    save_file_name = os.path.join(train_dir, save_file_name)
    check_file_exists(os.path.dirname(save_file_name))
    
    # Save model every epoch
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
    callbacks=[save_model]
    
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[0][1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[0][1], len(X_profiling[0])))
        sys.exit(-1)
        
    print ('[INFO]: Using train_model_dict')
    if val_split == 0:
        history = model.fit(x_train_dict, y_train_dict, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    else:
        print ('[INFO]: Validation split:', val_split)
        history = model.fit(x=Reshaped_X_profiling, y=tf.keras.utils.to_categorical(Y_profiling, num_classes=256), 
                            batch_size=64, epochs=5, validation_split=val_split)
    return history
#################################################################################