from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import numpy as np 
import matplotlib.pyplot as plt


def draw_roc(root):
	# input root
	# return np array of y and y_prob
	y, y_prob = [], []
	f = open(root, 'r')
	lines = f.readlines()
	for line in lines:
		l = line.strip().split('\t')
		y1, y2 = float(l[0]), float(l[1])
		y.append(y1); y_prob.append(y2)
	return np.array(y), np.array(y_prob)

if __name__ == '__main__':
	uni =  ['kki', 'caltech', 'pitt', 'usm', 'um', 'nyu', 'kul', 'ohsu', 'ucla', 'yale', 'mpg', 'olin', 'sjh', 'stanford', 'sbl', 'sdsu']
	fig = plt.figure(figsize=(4,4))

	#plt.title("ROC")

	# for u in uni:
	# 	root = 'abide_' + u+'.txt'
	# 	y, y_prob = draw_roc(root)
	# 	fpr, recall, thresholds = roc_curve(y, y_prob, pos_label=1)
	# 	roc_auc = auc(fpr, recall)
	# 	plt.plot(fpr, recall, label='AUC of %s = %.3f' % (u, roc_auc))
	# 	plt.plot([0,1], [0,1], 'r--')


	#uni = ['abide.txt', 'preschool_lasso.txt']
	uni = ['abide.txt']
	#uni = ['preschooler.txt']
	for u in uni:
		y, y_prob = draw_roc(u)
		fpr, recall, thresholds = roc_curve(y, y_prob, pos_label=1)
		roc_auc = auc(fpr, recall)
		mean_tpr = np.mean(recall, axis=0)
		std_tpr = np.std(recall, axis=0)
		upper = np.minimum(mean_tpr + std_tpr, 1)
		lower = np.maximum(mean_tpr - std_tpr, 0)

		plt.plot(fpr, recall, label='AUC of %s = %.3f' % (u[:-4], roc_auc))
		plt.plot([0,1], [0,1], 'r--')
		plt.annotate("%.2f (Sen:%.2f, Spe:%.2f)" % (0.50, 0.86, 0.62), (0.35, 0.80), xytext=(0.35,0.65), arrowprops=dict(facecolor='black', shrink=0.05))
		# plt.annotate("%.2f (Sen:%.2f, Spe:%.2f)" % (0.50, 0.88, 0.75), (0.26, 0.88), xytext=(0.30,0.75), arrowprops=dict(facecolor='black', shrink=0.05))

		#plt.fill_between(mean_tpr, lower, upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

		#print(['%.2f' % (threshold) for threshold in thresholds])
		for a,b,c in zip(fpr, recall, thresholds):
			print('%.2f\t%.2f\t%.2f' % (1-a,b, c))


	plt.legend(loc='lower right')
	plt.xlim([-.02, 1.02])
	plt.xticks(np.arange(0,1.1,0.2))
	plt.ylim([-.02, 1.02])
	plt.yticks(np.arange(0,1.1,0.2))
	plt.ylabel('Sensitivity')
	plt.xlabel('1-Specificity')
	plt.tight_layout()
	#plt.savefig('abide_uni.png')
	plt.savefig('abide_reori.svg')
	#plt.savefig("preschooler_test_reori.svg")