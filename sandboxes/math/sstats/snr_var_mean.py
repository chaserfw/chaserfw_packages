from sstats import snr_var_mean
from sfileengine import TRSFileEngine
from sfileengine import H5FileEngine
import matplotlib.pyplot as plt
from tqdm import trange

"""
fengine = TRSFileEngine(r'D:\SCA\datasets\PASCAD\ASCAD_EPF_20201129-151809.trs')#normal
fengine2 = TRSFileEngine(r'D:\SCA\datasets\PASCAD\ASCAD_EPF_20201129-151305.trs')#dilated
fengine5 = H5FileEngine(r'D:\ASCAD_EPF_20210108-111815.h5', group='/Profiling_traces')
fengine6 = H5FileEngine(r'D:\ASCAD_EPF_20210108-114638.h5', group='/Profiling_traces')
"""

all_fengines = []
all_fengines.append([H5FileEngine(r'D:\AES_PT\AES_PT_masked-D1-compressed-20210210-080100.h5', group='/Attack_traces'), 'compressed-20210210-080100'])
all_fengines.append([H5FileEngine(r'D:\AES_PT\AES_PT_masked-D1-compressed-20210205-132437.h5', group='/Attack_traces'), 'compressed-20210205-132437'])
all_fengines.append([H5FileEngine(r'D:\AES_PT\AES_PT_masked-D1-compressed_PT15005.h5', group='/Attack_traces'), 'compressed_PT15005'])
all_fengines.append([H5FileEngine(r'D:\AES_PT\AES_PT_masked-D1-compressed-pt199936.h5', group='/Attack_traces'), 'compressed-pt199936'])
all_fengines.append([H5FileEngine(r'D:\AES_PT\AES_PT_masked-D1-compressed-20210203-145733.h5', group='/Attack_traces'), 'compressed-20210203-145733'])

batch_index = 0
amount = 10000
times = 4
for j in trange (times):
    for i in trange(len(all_fengines)):
        snr = snr_var_mean(all_fengines[i][0], amount, batch_index)
        plt.plot(snr, color='r', lw=0.8)
        plt.title(all_fengines[i][1])
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('{}.pdf'.format(all_fengines[i][1] + '-' + str(batch_index) + '-' + str(amount)))
        plt.close()
    batch_index += amount


"""
plt.plot(snr, color='r', lw=0.8)
plt.title('SNR latent space ASCAD fixed - normal')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.plot(snr2, color='r', lw=0.8)
plt.title('SNR latent space ASCAD fixed - dilated')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.plot(snr3, color='r', lw=0.8)
plt.title('SNR noisy latent AES_PT')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.plot(snr4, color='r', lw=0.8)
plt.title('SNR less noisy latent AES_PT')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.plot(snr5, color='r', lw=0.8)
plt.title('SNR latent ASCAD random - dialted 1')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.plot(snr6, color='r', lw=0.8)
plt.title('SNR latent ASCAD random - dialted 2')
plt.tight_layout()
plt.grid(True)
plt.show()
"""