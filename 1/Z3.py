def GetTotalSalary(hours, hourlyPay):
	return float(hours) * float(hourlyPay)


workingHours = input("Dai working hours: ")
hourlySalary = input("Dai hourly salary: ")
print(GetTotalSalary(workingHours, hourlySalary))
