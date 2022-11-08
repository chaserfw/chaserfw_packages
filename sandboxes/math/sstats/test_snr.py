from sstats import snr_byte
import trsfile
trs_file_path = r'C:\Users\slpaguada\ChipWhisperer5_64\chipwhisperer\jupyter\Servio\traces_storaged\acq_20201008-141959\acq_20201008-141959-original.trs'
trs_file = trsfile.open(trs_file_path, mode='r')
print (dir(trs_file))
print (dir(trs_file.engine))
print (trs_file.engine.sample_length)
print (trs_file.engine.trace_length)
print (snr_byte(trs_file, 16))