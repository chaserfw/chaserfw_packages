from processors.perturbations.noise import add_gussian_noise
from sfileengine import H5FileEngine

profile = H5FileEngine(r'.\ASCAD_var_desync100-whole-clean-scaled.h5', group='/Profiling_traces')
attack = H5FileEngine(r'.\ASCAD_var_desync100-whole-clean-scaled.h5', group='/Attack_traces')

add_gussian_noise(profile, attack)
profile.close()
attack.close()