import regex as re

mboxFile = open("mbox.txt")
data = mboxFile.readlines()

sum = 0
count = 0
for line in data:
	if line.startswith("X-DSPAM-Confidence") == 0:
		sum += float(re.findall("\d+\.\d*", line)[0])
		count += 1

print(sum, count, sum / count)