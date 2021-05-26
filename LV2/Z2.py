import random

import numpy as np
import matplotlib.pyplot as plt

N = 1000

x = np.array([random.randint(1, 6) for i in range(N)])
y = [[a for a in x if a == i] for i in range(1, x.max() + 1)]
y = [len(a) for a in y]
# y = np.array([a for a in range(6)])
print(x, y)

# plt.axis([0, 7, 0, int(N / 3)])
# plt.bar([a for a in range(1, 7)], y, align="center")
plt.hist(x=x, density=True, rwidth=1, align="mid", bins=6)
plt.show()