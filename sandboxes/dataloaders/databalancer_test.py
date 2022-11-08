from dataloaders import DataBalancer
from dataloaders import DataLoader
from sfileengine import H5FileEngine
import os
from scrawler import SettingsCrawler
from collections import Counter

path = r'C:\Users\slpaguada\PythonProjects\Methodology-for-efficient-CNN-architectures-in-SCA\ASCAD\N0=0\ASCAD_dataset\ASCAD_dataset'
dataset_name = 'ASCAD.h5'
file = H5FileEngine(os.path.join(path, dataset_name), group='/Profiling_traces')
DataBalancer.create_index_map(file)
file.close()

print ('[INFO]: Load data balancer')
db = DataBalancer('index_map.pickle')
print (db.get_minor_class())

filename = 'attack'
file = H5FileEngine(os.path.join(path, dataset_name), group='/Attack_traces')
DataBalancer.create_index_map(file, filename, suffix=True)
file.close()

print ('[INFO]: Load data balancer (attack)')
db = DataBalancer('index_map_attack.pickle')
print (db.get_minor_class())

dataset_path = os.path.join(path, dataset_name)
balancer_file = 'index_map.pickle'
n_index = 100
n_index_val = 54
(X_profiling, Y_profiling, pro_index), (X_validation, Y_validation, val_index) = \
DataLoader.ProfileLoader.with_balancer(dataset_path, balancer_file, n_index, n_index_val, scalers_list=None)

print ('Located')
print (len([item for item, count in Counter(pro_index).items() if count > 1]))
print (pro_index)
for i, value in enumerate(pro_index):
    if 27204 == value:
        print (i)

print ('Located')
print (len([item for item, count in Counter(val_index).items() if count > 1]))
print (val_index)
for i, value in enumerate(val_index):
    if 47862 == value:
        print (i)

print ('---------------------------Attack')
dataset_path = os.path.join(path, dataset_name)
balancer_file = 'index_map_attack.pickle'
n_index = 100
(X_profiling, Y_profiling, plt_attack, att_index) = DataLoader.AttackLoader.with_balancer(dataset_path, balancer_file, n_index, scalers_list=None)

print (DataLoader.AttackLoader.LoaderMethod)
print ('Located')
print (len([item for item, count in Counter(att_index).items() if count > 1]))
print (att_index)
for i, value in enumerate(att_index):
    if 5850 == value:
        print (i)


settingCrawler = SettingsCrawler()
settingCrawler.from_dataloader()
settingCrawler.save_settings('./training_settings.json')