from sarithmetics import vector_addition_numpy

def remove_mean(trs_file, mean_vector=None):
	print ('[INFO *RemoveMean*]: Ready')
	if mean_vector is not None:
		print ('[INFO *RemoveMean*]: Removing mean using *VectorAdditionNumpy*')
		vector_addition_numpy(trs_file, mean_vector, sustract=True)
	else:
		print ('[INFO *RemoveMean*]: MeanVector None, computing mean using *MeanProcess*')
		from sstats import compute_samples_mean_trs
		n_traces = len(trs_file)
		mean_vector = compute_samples_mean_trs(trs_file, n_traces)
		vector_addition_numpy(trs_file, mean_vector, sustract=True)
	print ('[INFO *RemoveMean*]: Done')