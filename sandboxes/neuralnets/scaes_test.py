from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import tables
import os

from strainfuctions.scaes import SCAESGridSearch
from strainfuctions.scaes import SCAESPolicy

def CNN_N0(input_dim, n_channels=1, optimizer='RMSprop', classes=256):
    input_layer = tf.keras.Input(shape=(input_dim,), name='categorical_input')
    x = tf.keras.layers.Reshape(target_shape=(input_dim, n_channels), input_shape=(input_dim,))(input_layer)
    #10-3 antes
    x = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling1D(2, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)

    # Classification layer
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)

    prediction_score = tf.keras.layers.Dense(classes, name='prediction_score', activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=prediction_score, name='categorical_n0')
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def CNN_N1(input_dim, n_channels=1, optimizer='RMSprop', classes=256):
    input_layer = tf.keras.Input(shape=(input_dim,), name='categorical_input')
    x = tf.keras.layers.Reshape(target_shape=(input_dim, n_channels), input_shape=(input_dim,))(input_layer)
    #10-3 antes
    x = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling1D(2, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)

    # Classification layer
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)

    prediction_score = tf.keras.layers.Dense(classes, name='prediction_score', activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=prediction_score, name='categorical_n1')
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def CNN_N2(input_dim, n_channels=1, optimizer='RMSprop', classes=256):
    input_layer = tf.keras.Input(shape=(input_dim,), name='categorical_input')
    x = tf.keras.layers.Reshape(target_shape=(input_dim, n_channels), input_shape=(input_dim,))(input_layer)
    #10-3 antes
    x = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling1D(2, strides=2)(x)

    x = tf.keras.layers.Conv1D(16, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling1D()(x)

    x = tf.keras.layers.Flatten()(x)

    # Classification layer
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu')(x)

    prediction_score = tf.keras.layers.Dense(classes, name='prediction_score', activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=prediction_score, name='categorical_n2')
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def create_model(selector, input_dim, optimizer='RMSprop', n_channels=1, classes=256):
    if selector == 0:
        return CNN_N0(input_dim, n_channels, optimizer, classes)
    elif selector == 1:
        return CNN_N1(input_dim, n_channels, optimizer, classes)
    elif selector == 2:
        return CNN_N2(input_dim, n_channels, optimizer, classes)


dataset_model = '/home/bota2/datasets/datasets/ASCAD_dataset'
ascad = tables.open_file(os.path.join(dataset_model, 'ASCAD.h5'), mode='r')

profiling_traces = np.array(ascad.root.Profiling_traces.traces[:], dtype=np.int32)
scaler = MinMaxScaler()

# Prepare profiling data
profiling_traces = scaler.fit_transform(profiling_traces)
X = np.array(profiling_traces[:45000], dtype=np.float)
y = np.array(ascad.root.Profiling_traces.labels[:45000], dtype=np.uint8)
X_val = np.array(profiling_traces[45000:], dtype=np.float)
y_val = np.array(ascad.root.Profiling_traces.labels[45000:], dtype=np.uint8)

# Prepare attack data
attack_traces = scaler.transform(ascad.root.Attack_traces.traces[:10000])
plt_att = ascad.root.Attack_traces.metadata[:10000]['plaintext']
metadata = ascad.root.Attack_traces.metadata[:10000]
real_k = ascad.root.Attack_traces.metadata[0]['key'][2]
print ('[INFO]: real_k', real_k)
print ('[INFO]: closing file')
ascad.close()

# Initializing SCA-ES Policy
plt_attack        = plt_att
nb_traces_attacks = 5000
correct_key       = real_k
nb_attacks        = 1
attack_byte       = 2
es                = True
minimal_trace     = 100

sca_es_policy = SCAESPolicy(attack_traces, 
                            plt_attack, 
                            nb_traces_attacks, 
                            correct_key, 
                            nb_attacks, 
                            attack_byte,
                            es=es,
                            minimal_trace=minimal_trace)

# Defining parameters space
param_grid               = {}
param_grid['selector']   = [0, 1, 2]
param_grid['batch_size'] = [50, 100]
param_grid['epochs']     = [50, 100]
param_grid['optimizer']  = ['RMSprop', 'adam']
param_grid['input_dim']  = [700]

# Create classifier chooser using KerasClassifer wrapper
classifier_choser = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, **param_grid)
print (classifier_choser.get_params())

sca_es_grid_search = SCAESGridSearch(classifier_choser, save_ge_logs=True)
#sca_es_grid_search.X_set = X
#sca_es_grid_search.y_set = tf.keras.utils.to_categorical(y, num_classes=256)
#sca_es_grid_search.val_X_set = X_val
#sca_es_grid_search.val_y_set = tf.keras.utils.to_categorical(y_val, num_classes=256)

# Prepare callbacks
#save_file_name = os.path.join('./GridSearchResult', 'm{}_{}.h5'.format(1, 2))
#save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)
callbacks=[sca_es_policy]
sca_es_grid_search.set_auto_checkpoint_callback('./GridSearchResult')

# Fit model
grid_result = sca_es_grid_search.fit(x=X, y=tf.keras.utils.to_categorical(y, num_classes=256), 
                                     callbacks=callbacks, 
                                     validation_data=(X_val, tf.keras.utils.to_categorical(y_val, num_classes=256)),
                                     epochs=10)