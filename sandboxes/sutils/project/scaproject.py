from sutils.project.sca import SCAProject

dataset_path = '/home/bota2/datasets/datasets/ASCAD_dataset'
root_project = './root_project'
scaProject = SCAProject('Adaptability', root_project, id_train='20211110-144949')
scaProject.add_dataset_path(dataset_path)
scaProject.update()
#dataset_path = 'drive/My Drive/datasets/ASCAD_noisy_level_1.h5'
#meta_map = {'plt':2, 'key':34, 'mas':50} #ASCAD_F

#dataset_path = 'drive/My Drive/datasets/ASCAD_var_desync100.h5'
#meta_map = {'plt':2, 'key':18, 'mas':34} #ASCAD_R

#file_dataset = tables.open_file(dataset_path, 'r')
#real_key     = file_dataset.root.Attack_traces.metadata[0]['key']
#file_dataset.close()
#correct_key = real_key[2]

#compress_name = '{}-compress.h5'.format(dataset_path.split(os.path.sep)[-1][:-3])

print ('Train dir', scaProject.TrainDir)
print ('ID train', scaProject.TrainID)
print ('Method path', scaProject.ProjectRootPath)
print ('Method path destination', scaProject.MethodPathDestination)
print (scaProject.get_dataset_path())