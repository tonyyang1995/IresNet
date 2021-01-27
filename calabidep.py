import os
from scipy import stats
import statistics

# a = open('csv/abide_test.csv', 'r')
# lines = a.readlines()



# head = lines[0].strip().split(',')

# for j in range(2, 170):
# 	asd = []
# 	td = []
# 	for line in lines[1:]:
# 		conts = line.strip().split(',')
# 		name = conts[0]
# 		v = float(conts[j])
# 		if 'ASDP' in name:
# 			asd.append(v)
# 		else:
# 			td.append(v)

# 	t, p = stats.ttest_ind(asd, td)
# 	if p < 0.05:
# 		print(head[j], p)

a = open('csv/abide_train_test.csv', 'r')
lines = a.readlines()
head = lines[0].strip().split(',')

for j in range(14, 14+166):
	asd = []
	td = []
	for line in lines[1:]:
		conts = line.strip().split(',')
		group = conts[1]
		v = float(conts[j])
		if group == 'TD':
			td.append(v)
		else:
			asd.append(v)
	t, p = stats.ttest_ind(asd, td)
	print(head[j], p)