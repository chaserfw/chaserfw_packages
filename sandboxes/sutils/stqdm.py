from sutils import trange
from sutils import tqdm

#for i in trange(10, desc='Hola'):
    #pass

pbar1 = tqdm(total=10, position=0, leave='output')
for i in range(10):
    pbar1.update()