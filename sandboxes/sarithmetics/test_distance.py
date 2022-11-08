from sarithmetics import euclidean_linalg
from sarithmetics import euclidean_scipy
from sarithmetics import euclidean_distance
import numpy as np
import scipy.spatial as ss

vectors = np.random.randint(0, 100, size=(10, 100))
x = np.array([[0], [1], [0]])
y = np.array([[0], [0], [1]])
print (euclidean_linalg(x, y))
print (euclidean_scipy(x, y))
print (euclidean_distance(x, y))

x_y = [[0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]
#x_y = [[1.3],[3.7],[5.1],[2.4]]
tree = ss.cKDTree(x_y) # c de continous?
print (tree)
print (dir(tree))

k = 3
intens=1e-10
print (x_y)
x_y = [list(p + intens*np.random.rand(len(x_y[0]))) for p in x_y]
print (x_y)
nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x_y]
print (nn)
print (np.log(nn))

'''
273ms

265ms

266ms

258ms

259ms

254ms

255ms

257ms

261ms

257ms

266ms

263ms

267ms

260ms

259ms

256ms

258ms

261ms

257ms

254ms

266ms

257ms

254ms

257ms

257ms

260ms

261ms

264ms

262ms

262ms

280ms

265ms

264ms

261ms

256ms

259ms

262ms

256ms

257ms

267ms

266ms

264ms

262ms

263ms

259ms

265ms

259ms

262ms

261ms

257ms

261ms

259ms

261ms

263ms

259ms

259ms

255ms

257ms

262ms

254ms

260ms

262ms

264ms

260ms

260ms

259ms

261ms

264ms

262ms

256ms

260ms

258ms

255ms

262ms

256ms

261ms

256ms

258ms

262ms

260ms

263ms

264ms

267ms

269ms

268ms

266ms

279ms

269ms

261ms

260ms

267ms

263ms

267ms

274ms

280ms

267ms

270ms

270ms

278ms

275ms

'''