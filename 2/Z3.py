import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1, 2, 3, 4, 5, 6), delimiter=",", skiprows=1)
# print(data)
mpg = np.array([a[0] for a in data])
hp = np.array([a[3] for a in data])
wt = np.array([a[5] for a in data])
wt = wt * 30

print(mpg, hp)

plt.xlabel("mpg")
plt.ylabel("hp")

plt.scatter(mpg, hp, wt)

# 6 cylinders
mpg = np.array([a[0] for a in data if a[1] == 6])
hp = np.array([a[3] for a in data if a[1] == 6])
wt = np.array([a[5] for a in data if a[1] == 6])
wt = wt * 30

# plt.scatter(mpg, hp, wt)

plt.show()
