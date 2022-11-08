import sys
def ipython_info():
	ip = False
	if 'ipykernel' in sys.modules:
		ip = 'notebook'
	elif 'IPython' in sys.modules:
		ip = 'terminal'
	return ip

env = ipython_info()
if env == 'terminal' or not env:
	from tqdm import tqdm as rtqdm
	from tqdm import trange as rtrange
else:
	from tqdm.notebook import tqdm as rtqdm
	from tqdm.notebook import trange as rtrange

def trange(arg, **kwargs):
	return rtrange(arg, **kwargs)

def tqdm(total, **kwargs):
	return rtqdm(total, **kwargs)