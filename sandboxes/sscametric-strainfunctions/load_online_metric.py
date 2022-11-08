import tensorflow as tf
from sscametric import onlinemetrics
from dataloaders import load_attack_dataset_sss
from dataloaders import load_dataset_sss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from strainfuctions import custom
import numpy as np
from sutils import save_scaler

class NNModels():
	def CNN_ZaidN0(self, classes, n_features, n_channels=1, lr=0.005, add_metrics=[]):
		# Personal design
		input_layer = tf.keras.Input(shape=(n_features, n_channels), name='input_1')
		#10-3 antes
		x = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', 
								   padding='same', name='block1_conv1')(input_layer)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block1_pool')(x)
		
		x = tf.keras.layers.Flatten(name='flatten')(x)

		# Classification layer
		x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
		x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
		x = tf.keras.layers.Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

		# Logits layer
		x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

		# Create model
		model = tf.keras.Model(input_layer, x, name='ascad')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

		# add metric
		custom_metric = []
		custom_metric.append('accuracy')
		for metric in add_metrics:
			custom_metric.append(metric)

		custom_loss = custom.GECatCrossentropy(classes)
		model.compile(loss=custom_loss.loss_function, optimizer=optimizer, metrics=custom_metric)
		return model

#==================================================================
#
#==================================================================
scalers_list = [StandardScaler(), MinMaxScaler()]
dataset_path = r'.\ASCAD_dataset\ASCAD.h5'
train_dir = '.\\'
classes = 256
real_key = np.load(r'.\ASCAD_dataset\key.npy')
#==================================================================
#
#==================================================================
correct_key = real_key[2]
nb_traces_attacks = 2000
nb_attacks = 3
metric = onlinemetrics.key_rank_Metric(correct_key, nb_traces_attacks, nb_attacks)
print (metric)

print ('===============================load_dataset_sss')
n_train_split = 175
profile_limit = 47000
X_profiling, Y_profiling, index = load_dataset_sss(dataset_path, n_train_split, train_limit=profile_limit, scalers_list=scalers_list)
print (X_profiling[0:5])
print (Y_profiling[0:5])
print (index[0:5])

print ('===============================load_attack_dataset_sss')
n_attack_split = 12
attack_limit = 20000
# Dentro se aplican los escaler, si estos ya estan fit, sino se fit y luego se aplican
X_attack, Y_attack, plt_attack, index = load_attack_dataset_sss(dataset_path, n_attack_split, attack_limit=attack_limit, 
																scalers_list=scalers_list)


print (X_attack[0:5])
print (Y_attack[0:5])
print (plt_attack[0:5])
print (index[0:5])

# Save scalers
for i, scaler in enumerate(scalers_list):
	save_scaler(path='scaler_{}'.format(i), scaler=scaler)

#==================================================================
#
#==================================================================
plt_profiling = np.random.randint(0, 256, size=(X_profiling.shape[0], plt_attack.shape[1]), dtype=np.uint8)
print ('antes:')
print (Y_profiling.shape)
print (plt_profiling.shape)
print ('*------')
print (tf.keras.utils.to_categorical(Y_profiling, num_classes=256).shape)
print (np.concatenate((tf.keras.utils.to_categorical(Y_profiling, num_classes=classes), np.zeros(shape=(len(plt_profiling), 1)), plt_profiling), axis=1).astype(np.uint8).shape)
# prepare the label: label+pleintext (for metric calculation)
Y_profiling = np.concatenate((tf.keras.utils.to_categorical(Y_profiling, num_classes=classes), np.zeros(shape=(len(plt_profiling), 1)), plt_profiling), axis=1).astype(np.uint8)
Y_attack = np.concatenate((tf.keras.utils.to_categorical(Y_attack, num_classes=classes), np.ones(shape=(len(plt_attack), 1)), plt_attack), axis=1).astype(np.uint8)
print (Y_profiling.shape)
print ('----')
print (Y_attack.shape)
print ('----')
print (Y_attack[0][0:classes])
print (Y_attack[0][classes:])

X_profiling_reshaped = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
X_validation_reshaped = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

x_train_dict = {'input_1': X_profiling_reshaped}
y_train_dict = {'predictions': Y_profiling}

x_validation_dict = {'input_1': X_validation_reshaped}
y_validation_dict = {'predictions': Y_attack}
#==================================================================
#
#==================================================================
nnModel    = NNModels()
model      = nnModel.CNN_ZaidN0(256, 700, add_metrics=[metric])
epochs     = 50
batch_size = 50
maxlr      = 0.005
fname      = 'cnn.h5'

trainerVal = custom.GETrainerValNN(model, epochs=epochs, batch_size=batch_size, train_dir=train_dir, 
						  maxlr=maxlr, sfilename=fname)
trainerVal.XTrainDict = x_train_dict
trainerVal.YTrainDict = y_train_dict
trainerVal.XValDict   = x_validation_dict
trainerVal.YValDict   = y_validation_dict
trainerVal.OCPsps     = x_train_dict['input_1'].shape[0]
trainerVal.train()