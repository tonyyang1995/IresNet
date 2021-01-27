import os
from scipy import stats
import statistics

a = open('csv/ours_train_test.csv', 'r')
lines = a.readlines()

head = lines[0].strip().split(',')

for j in range(28, 28+166):
	asd = []
	td = []
	for line in lines[1:]:
		conts = line.strip().split(',')
		name = conts[0]
		v = float(conts[j])
		if 'ASDP' in name:
			asd.append(v)
		else:
			td.append(v)

	t, p = stats.ttest_ind(asd, td)
	#if p < 0.05:
	print(head[j], p)