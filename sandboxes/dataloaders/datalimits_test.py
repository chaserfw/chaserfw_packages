from dataloaders import DataLoader
import os
from scrawler import SettingsCrawler
from collections import Counter

path = r'C:\Users\slpaguada\PythonProjects\Methodology-for-efficient-CNN-architectures-in-SCA\ASCAD\N0=0\ASCAD_dataset\ASCAD_dataset'
dataset_name = 'ASCAD.h5'

dataset_path = os.path.join(path, dataset_name)
(X_profiling, Y_profiling, pro_index), (X_validation, Y_validation, val_index) = \
DataLoader.ProfileLoader.with_limiters(dataset_path, train_limit=45000, val_limit=50000, scalers_list=None)

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
(X_profiling, Y_profiling, plt_attack, att_index) = DataLoader.AttackLoader.with_limiters(dataset_path, 20000, scalers_list=None)

print (DataLoader.AttackLoader.LoaderMethod)
print ('Located')
print (len([item for item, count in Counter(att_index).items() if count > 1]))
print (att_index)
for i, value in enumerate(att_index):
    if 5850 == value:
        print (i)

print(plt_attack[5850])
settingCrawler = SettingsCrawler()
settingCrawler.from_dataloader()
settingCrawler.save_settings('./training_settings_limiters.json')