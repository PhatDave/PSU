def determineGrade(score):
	try:
		score = float(score)
		if score > 1:
			return 'N/A'
		elif score >= 0.9:
			return 'A'
		elif score >= 0.8:
			return 'B'
		elif score >= 0.7:
			return 'C'
		elif score >= 0.6:
			return 'D'
		elif score >= 0:
			return 'F'
		else:
			return 'N/A'
	except (TypeError, ValueError):
		return 'Err'


print(determineGrade(input("Dai score: ")))
