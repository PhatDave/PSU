import requests as re
import regex as r
import pandas as pd
import datetime
from calendar import monthrange
import matplotlib.pyplot as plt

text = ""
file = None
try:
	file = open("cache.txt", 'r')
	text = text.join(file.readlines())
except IOError:
	site = re.get(
		"http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=5&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017")
	file = open("cache.txt", 'w')
	file.write(site.text)
	text = site.text
finally:
	file.close()

data = r.findall("<Podatak><vrijednost>(\d+\.?\d*)</vrijednost>", text)
data = [float(data[i]) for i in range(data.__len__())]
dataTime = r.findall("<vrijeme>([0-9-:T+]+)</vrijeme>", text)
newDataTime = []
months = []
days = []
for date in dataTime:
	temp = r.findall("(\d+)\-(\d+)\-(\d+)", date)
	datetimeObject = datetime.datetime(int(temp[0][0]), int(temp[0][1]), int(temp[0][2]))
	months.append(int(temp[0][1]))
	days.append(int(temp[0][2]))
	newDataTime.append(datetimeObject)

data = {'val': data, 'time': dataTime, 'date': newDataTime, 'month': months, 'day': days}
data = pd.DataFrame(data)

print(data.sort_values('val').tail(3))

# missingData = {}
# for i in range(1, 13):
# 	missingData[i] = monthrange(newDataTime[0].year, i)[1]
# 	missingData[i] -= data[data.month == i].__len__()
# plt.bar(x=[i for i in missingData], height=[missingData[i] for i in missingData])
# plt.xticks([i for i in missingData])
# plt.show()

# plt.boxplot(data[data.month == 1].val, positions=[1])
# plt.boxplot(data[data.month == 6].val, positions=[6])
# plt.show()

