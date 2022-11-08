import numpy as np

# This success rate works weard, better check before use it
def success_rate(predictions:np.ndarray, target_labels:np.ndarray):
	#Compute succes rate
	span = range(1, target_labels.max()-target_labels.min()+1, 1)
	c_sr = np.empty(len(span))

	n_traces = predictions.shape[0]
	for order in span:
		succes = np.zeros(n_traces)
		for k in range(n_traces):
			for o in range(order):
				if target_labels[k] == predictions[k][o]:
					succes[k] = 1
		success_accuracy = np.sum(succes)/n_traces
		print("---success accuracy of order {} for one nibble = {}".format(order, success_accuracy))
		c_sr[order-1] = success_accuracy

	return span, c_sr