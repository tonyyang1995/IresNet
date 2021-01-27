import numpy as np
import matplotlib.pyplot as plt 

def load_txt(name):
	f = open(name)
	lines = f.readlines()
	y = []
	y_prob = []
	for line in lines:
		l = line.strip().split('\t')
		y1, y2 = float(l[0], l[1])
		y.append(y1)
		y_prob.append(y2)

	f.close()
	return np.array(y), np.array(y_prob)

def draw_our_box():
	f = open('preschool_lasso1.txt')
	lines = f.readlines()
	ys = []
	for line in lines:
		l = line.strip().split('\t')
		y_td, y_asd, pred_td, pred_asd = l
		#print(pred_td, pred_asd)
		pred_td, pred_asd = float(pred_td), float(pred_asd)
		if pred_td > pred_asd: # draw with different color
			ys.append([pred_asd, 0]) # use the axis 1 probability
		else:
			ys.append([pred_asd, 1])
	#print(ys)
	ys = sorted(ys, key=lambda x:x[0])
	#print('after')
	#print(ys)
	x = [i for i in ys]

	asd = []; x_asd = []
	td = []; x_td = []
	for i, v in enumerate(ys):
		if v[1] == 0:
			td.append(v[0])
			x_td.append(i)
		else:
			asd.append(v[0])
			x_asd.append(i)

	plt.figure(figsize=(4,4))
	ax = plt.gca()
	#ax.xaxis.set_ticks_position('center')
	ax.spines['bottom'].set_position('center')
	width =0.2
	rect = ax.bar(x_td, td, width, label='td')
	rect2 = ax.bar(x_asd, asd, width, label='asd')
	f.close()
	plt.tight_layout()
	plt.legend()

	plt.show()


if __name__ == '__main__':
	draw_our_box()
