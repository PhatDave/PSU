import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests as re

data = pd.read_csv('mtcars.csv')
print(data)

data.set_index('cyl').mpg.plot.bar(x=data.cyl)
plt.show()

data.wt.plot.bar()
plt.show()

plt.bar(x='0', height=data[data.am == 0].mpg.mean())
plt.bar(x='1', height=data[data.am == 1].mpg.mean())
plt.show()

plt.plot('hp', 'qsec', data=data[data.am == 0].sort_values('hp'), color='red')
plt.plot('hp', 'qsec', data=data[data.am == 1].sort_values('hp'), color='green')
plt.legend()
plt.show()