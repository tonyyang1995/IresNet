a = open('abide.txt', 'r')
lines = a.readlines()
b = open('data1.csv', 'w')
for line in lines:
	l = line.strip().split('\t')
	label = int(l[0])
	pred = float(l[1])
	if label == 0:
		b.write('%.4f,TD\n'%(pred))
	else:
		b.write('%.4f,ASD\n'%(pred))
