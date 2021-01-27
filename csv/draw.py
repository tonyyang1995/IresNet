import matplotlib.pyplot as plt

# read csv
f = open('ba_weight.csv', 'r')

lines = f.readlines()
head = lines[0]
head = head.strip().split(',')
head = head[1:]

ours = lines[1]
abide = lines[2]
ours = ours.strip().split(',')[1:]
abide = abide.strip().split(',')[1:]

x = [i for i in range(len(ours))]
ours = [float(i) for i in ours]
abide = [float(i) for i in abide]

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

ax.scatter(x, ours, alpha=.5, marker='o', label='Preschooller', s=2, c='blue')
ax.scatter(x, abide, alpha=.5, marker='o', label='Abide', s=2, c='red')

# ax.set_ylim([0, 0.1])
ax.set_yticks([0.01, 0.05])

ax.legend(fontsize=10)
plt.savefig('pvalues.png')
plt.close('all')
