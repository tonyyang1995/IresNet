import os
labels = {}

f = open('../dataset/abide/labels/test7.txt','r')
lines = f.readlines()
for line in lines:
    conts = line.strip().split('\t')
    name = conts[0].split('/')[-2]
    gt = conts[2]
    labels[name] = gt
f.close()

f = open('../dataset/abide/labels/train7.txt', 'r')
lines = f.readlines()
for line in lines:
    conts = line.strip().split('\t')
    name = conts[0].split('/')[-2]
    gt = conts[2]
    labels[name] = gt
f.close()

f = open('feats.csv', 'w')
a = open('ours_train.csv', 'r')
b = open('ours_test.csv', 'r')

lines = a.readlines()
header = lines[0].strip().split(',')
header = [header[0]] + ['type', 'label'] + header[1:]
mes = ','.join(header)
f.write(mes+'\n')

for line in lines[1:]:
    conts = line.strip().split(',')
    name = conts[0].split('/')[-1]
    mes = [name, 'train']
    if 'ASDP' in name:
        mes += ['1']
    elif 'ASDN' in name:
        mes += ['0']
    else:
        mes += ['']
    mes += conts[1:]
    mess = ','.join(mes)
    f.write(mess+'\n')

lines = b.readlines()
for line in lines[1:]:
    conts = line.strip().split(',')
    name = conts[0].split('/')[-1]
    mes = [name, 'test']
    if 'ASDP' in name:
        mes += ['1']
    elif 'ASDN' in name:
        mes += ['0']
    else:
        mes += ['']
    mes += conts[1:]
    mess = ','.join(mes)
    f.write(mess+'\n')

a.close()
b.close()

c = open('abide_train.csv', 'r')
d = open('abide_test.csv', 'r')
lines = c.readlines()
for line in lines[1:]:
    conts = line.strip().split(',')
    #print(conts)
    name = conts[0].split('/')[-1]
    mes = [name, 'train']
    label = labels[name]
    mes += [label]
    mes += conts[1:]
    mess = ','.join(mes)
    f.write(mess+'\n')

lines = d.readlines()
for line in lines[1:]:
    conts = line.strip().split(',')
    name = conts[0].split('/')[-1]
    mes = [name, 'test']
    label = labels[name]
    mes += [label]
    mes += conts[1:]
    mess = ','.join(mes)
    f.write(mess+'\n')

c.close()
d.close()
f.close()
