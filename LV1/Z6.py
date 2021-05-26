mboxFile = open("mbox-short.txt")
data = mboxFile.readlines()

emails = []
hostnames = {}
for line in data:
	if line.startswith("From: "):
		splitLine = line.split("From: ")[1].split("\n")
		try:
			emails.index(splitLine[0])
		except ValueError:
			emails.append(splitLine[0])

		splitLine = splitLine[0].split("@")
		if splitLine[1] in hostnames.keys():
			hostnames[splitLine[1]] += 1
		else:
			hostnames[splitLine[1]] = 1

print(emails)
print(sorted(hostnames.items(), key=lambda k: k[1]))
