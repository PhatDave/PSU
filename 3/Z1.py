import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('mtcars.csv')
print(data)
# print(mtcars[(mtcars.cyl == 4) & (mtcars.hp > 100)].car)

print(data.sort_values('mpg')[:-6:-1])
# print(data.sort_values('mpg').tail(5))

print(data[(data.cyl == 8)].sort_values('mpg')[:3])
# print(data[(data.cyl == 8)].sort_values('mpg').head(3))

print(data[(data.cyl == 6)].mpg.mean())

print(data[(data.cyl == 4) & (data.wt <= 2.2) & (data.wt >= 2.0)].mpg.mean())

print(str(data[data.am == 0].__len__()) + " manual, " + str(data[data.am == 1].__len__()) + " automatic")

print(str(data[(data.am == 1) & (data.hp > 100)].__len__()) + " automatic & hp > 100")

print(data.wt * 1000)