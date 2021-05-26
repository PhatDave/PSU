userInputs = []
while True:
	userInput = input("Dai  ")
	if userInput == "Done":
		break
	else:
		try:
			inputInteger = int(userInput)
			userInputs.append(inputInteger)
		except ValueError:
			print("Please enter integers")
			continue

sum = 0
avg = 0
min = 0
max = 1e90
for i in userInputs:
	t = float(i)
	sum += t
	if i < min:
		min = i
	if i > max:
		max = i
avg = sum / userInputs.__len__()

print(sum, avg, min, max)