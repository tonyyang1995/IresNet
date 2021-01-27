import numpy as np 
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

def setBoxColors(bp, color='blue'):
	plt.setp(bp['boxes'], color=color)
	plt.setp(bp['whiskers'], color=color)
	plt.setp(bp['caps'], color=color)
	plt.setp(bp['medians'], color=color)
	plt.setp(bp['fliers'], color=color, marker='+')

names = ['Rolandic_Oper_R', 'SupraMarginal_R', 'Precentral_R', 'Frontal_Inf_Oper_R', 'Heschl_R', 'Postcentral_R', 'Temporal_Sup_R', 'Frontal_Med_Orb_L', 'Frontal_Inf_Orb_2_L']
#abide: 27, 79, 15, 21, 93, 73, 95, 34, 24

# draw preschooler
#a = open('../csv/abide_train_test.csv', 'r')
a = open('../csv/ours_train_test.csv', 'r')

dicts = {}
feats_td = {}
feats_asd = {}
line = a.readline()
line = line.strip().split(',')
for name in names:
	#print(name, line.index(name))
	dicts[line.index(name)] = name
	feats_td[line.index(name)] = []
	feats_asd[line.index(name)] = []

lines = a.readlines()
for line in lines:
	line = line.strip().split(',')
	t = line[2]
	label = line[1]

	if t == 'testing':
		if label == 'TD':
			for k,v in feats_td.items():
				feats_td[k].append(float(line[k]))
		else:
			for k,v in feats_asd.items():
				feats_asd[k].append(float(line[k]))

pos = 1
fig = figure(figsize=(10,4))
ax = axes()
#hold(True)

for k,v in feats_asd.items():
	v1 = feats_td[k]
	A = [v1]
	B = [v]
	bp = boxplot(A, positions=[pos], widths = 0.8, sym='lime', whis=1.5, labels=['TD'])
	setBoxColors(bp, color='lime')
	bp1 = boxplot(B, positions=[pos+1], widths = 0.8, sym='r', whis=1.5, labels=['ASD'])
	setBoxColors(bp1, color='red')
	plt.legend([bp1['boxes'][0], bp['boxes'][0]], ['ASD', 'TD'], fontsize=6)

	pos += 3

ax.set_xticklabels(names, rotation=60)
las = []
temp = 1.5
for i in range(0, 9):
	las.append(temp)
	temp += 3

ax.set_xticks(las)
plt.tight_layout()
plt.ylim([0.2, 1.0])
plt.savefig('our_boxplot2.png')