import tables

path = r'D:\cw_datasets\20200430-120909\20200430-120909-sep-1800.h5'
h5 = tables.open_file(path, mode='r+')

h5.rename_node('/Profiling_traces/label', 'labels')
h5.close()