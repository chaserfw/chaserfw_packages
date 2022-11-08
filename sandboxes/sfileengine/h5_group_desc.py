from sfileengine import h5_group_descriptor
import tables
#path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'
path = r'D:\cw_datasets\20200430-120909\ascad_acq_20200430-120909.h5'
h5 = tables.open_file(path)
print (help(h5))
#print (h5.get_node('/traces'))


#print (h5_group_descriptor(path))