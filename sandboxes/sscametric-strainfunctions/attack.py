from sscametric import perform_attacks
import tensorflow as tf
import tables
import numpy as np
import matplotlib.pyplot as plt
from sutils import load_scaler

file = tables.open_file(r'.\ASCAD_dataset\ASCAD.h5', mode='r')
real_key = np.load(r'.\ASCAD_dataset\key.npy')
X_attack = file.root.Attack_traces.traces[0:50000]
plt_attack = file.root.Attack_traces.metadata[0:50000]['plaintext']
nb_attacks = 10
nb_traces = 3000
model = tf.keras.models.load_model(r'.\cnn-3\cnn.h5', compile=False)

# Load and apply scalers
for i in range(2):
	scaler = load_scaler(path=r'.\cnn-3\scaler_{}'.format(i))
	X_attack = scaler.transform(X_attack)

predictions = model.predict(X_attack)
avg_rank = np.array(perform_attacks(nb_traces, predictions, plt_attack, correct_key=real_key[2], 
									nb_attacks=nb_attacks, output_rank=True, pbar=True))
plt.plot(avg_rank)
plt.tight_layout()
plt.savefig('guessing_entropy.pdf')
plt.close()
file.close()