import numpy as np
import matplotlib.pyplot as plt
import math


def deder(s=32, n=16):
	b = np.array([0] * (s * s))
	b.shape = (s, s)
	# b.shape = (int(math.sqrt(s)), int(math.sqrt(s)))
	w = b + 255
	r = [w, b]

	r1 = np.hstack([(lambda x: r[x % 2])(a) for a in range(n)])
	r2 = np.hstack([(lambda x: r[(x - 1) % 2])(a) for a in range(n)])
	r = [r1, r2]

	img = np.vstack([(lambda x: r[(x - 1) % 2])(a) for a in range(n)])
	plt.imshow(np.array(img), cmap='gray', vmin=0, vmax=255)
	plt.show()


deder(4, 4)
