import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(root):
	f = open(root, 'r')
	lines = f.readlines()
	xs, asdns, asdps = [], [], []
	for line in lines:
		a = line.strip().split('\t')
		x, asdn, asdp = int(a[0]), float(a[1]), float(a[2])
		xs.append(x)
		asdns.append(asdn)
		asdps.append(asdp)
	return xs, asdns, asdps

if __name__ == '__main__':
	# roots = ['ours_p_value_reori_axial.txt', 'ours_p_value_reori_coronal.txt', 'ours_p_value_reori_sagittal.txt']
	roots = ['abide_p_value_axial.txt', 'abide_p_value_coronal.txt', 'abide_p_value_sagittal.txt']
	for root in roots:
		x, y1, y2 = load_data(root)
		name = root[:-4] + '.png'
		pos = root[:-4].split('_')[-1]
		f1 = np.polyfit(x, y1, 1)
		f2 = np.polyfit(x, y2, 1)
		p1 = np.poly1d(f1)
		p2 = np.poly1d(f2)

		y_fit1 = p1(x)
		y_fit2 = p2(x)

		fig = plt.figure(figsize=(6,4))
		ax = fig.add_subplot(1,1,1)

		ax.scatter(x, y1, alpha=.5, s=2, marker='o')
		ax.scatter(x, y2, alpha=.5, s=2, marker='o')
		ax.plot(x, y_fit1, label='TD in '+pos + 'MRI', markersize=2, c="blue")
		ax.plot(x, y_fit2, label='ASD in '+pos + 'MRI', markersize=2, c="red")

		ax.set_xlabel('Different Slice on ' + pos + ' Area', fontsize=10)
		ax.set_ylabel('mean of RNLY Model Value Per Slice', fontsize=10)

		ax.legend(fontsize=10)
		plt.savefig(name)
		plt.close('all')

