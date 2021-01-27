import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

class LassoNet(nn.Module):
    def __init__(self):
        super(LassoNet, self).__init__()
        self.fc1 = nn.Linear(166, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        return x
    
class csvDataset(torch.utils.data.Dataset):
    def __init__(self, label_type, dt=None):
        f = open('csv/feats.csv', 'r')
        lines = f.readlines()
        self.y = []
        self.x = []
        self.names = []
        for line in lines[1:]:
            conts = line.strip().split(',')
            t = conts[1]
            if t == label_type:
                if dt is None:
                    label = float(conts[2])
                    self.y.append(label)
                    feats = [float(x) for x in conts[3:]]
                    self.x.append(feats)
                    self.names.append(conts[0])
                elif dt in conts[0]:
                    label = float(conts[2])
                    self.y.append(label)
                    feats = [float(x) for x in conts[3:]]
                    self.x.append(feats)
                    self.names.append(conts[0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        y = self.y[index]
        x = self.x[index]
        name = self.names[index]
        if y == 1:
            label = [0,1]
        else:
            label = [1,0]
        y = torch.FloatTensor(label)
        x = torch.FloatTensor(x)
        return x, y, name

def train(batch_size=8, lr=1e-3, epochs=200, dataroot='csv/feat.csv', label_type='train', dt=None):
    dataset = csvDataset(label_type, dt=dt)
    dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True
	)

    model = LassoNet().cuda()
    state_dict = torch.load('checkpoints/lasso/lasso_best.pth')
    model.load_state_dict(state_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.MSELoss()
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for i, (x_data, y_data, name) in enumerate(dataloader):
            x_data = Variable(x_data).cuda()
            y_data = Variable(y_data).cuda()
            optimizer.zero_grad()
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('epochs: %d\tLoss: %.4f' %(epoch, loss.item()))
                print('best_acc: %.4f' %(best_acc))
        
        path = os.path.join('checkpoints/lasso/lasso_%d.pth'%(epoch))
        torch.save(model.state_dict(), path)
        acc = test(save_epoch=epoch, dt=dt)
        if best_acc < acc:
            best_acc = acc
            path = os.path.join('checkpoints/lasso/lasso_best.pth')
            torch.save(model.state_dict(), path)



def test(batch_size=1, dataroot='csv/feat.csv', label_type='test', save_epoch=0, dt=None):
    #f = open('csv/ours_train.txt', 'w')
    f = open('roc_data/data2.csv', 'w')
    dataset = csvDataset(label_type=label_type, dt=dt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = LassoNet().cuda()
    if 'best' in save_epoch:
        model_path = 'checkpoints/preschooler/lasso_%s.pth'%(save_epoch)
    else:
        model_path = 'checkpoints/preschooler/lasso_%d.pth'%(save_epoch)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    TP, TN, FP, FN = 0,0,0,0
    total = len(dataloader)
    #print(total)
    for i, (x_data, y_data, name) in enumerate(dataloader):
        x_data = Variable(x_data).cuda()
        y_data = Variable(y_data).cuda()
        y_pred = model(x_data)
        output = torch.argmax(y_pred)
        gt = torch.argmax(y_data)
        #output2 = y_pred.sigmoid()
        #print(float(y_data[0,1]), y_pred[0,1])
        #rows = '%.4f\t%.4f\t%.4f\t%.4f\n' % (y_data[0,0], y_data[0,1], y_pred[0,0], y_pred[0,1])
        #rows = '%d,%.4f\n' % (int(output), float(y_pred[0,1]))
        #f.write(rows)
        if gt == 0:
            f.write('%.4f,TD\n'%(float(y_pred[0,1])))
        elif gt == 1:
            f.write('%.4f,ASD\n'%(float(y_pred[0,1])))

        name = name[0]
        mes = '%s,%d,%d'%(name, gt, output)
        #f.write(mes+'\n')

        if output.data == 1 and gt.data == 1:
            TP += 1
        elif output.data == 0 and gt.data == 0:
            TN += 1
        elif output.data == 1 and gt.data == 0:
            FP += 1
        elif output.data == 0 and gt.data == 1:
            FN += 1
    P = float(TP) / (TP + FP)
    R = float(TP) / (TP + FN)
    F1 = (2*P*R) / (P + R)
    sen = float(TP) / (TP + FN)
    spe = float(TN) / (TN + FP)
    acc = (float(TP) + float(TN))/ len(dataloader) * 100.0
    
    message = '\n------------------------results----------------------\n'
    message += '{:>10}\t{:>10}\n'.format('TP:', TP)
    message += '{:>10}\t{:>10}\n'.format('TN:', TN)
    message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
    message += '{:>10}\t{:>10.4f}\n'.format('precision:', P)
    message += '{:>10}\t{:>10.4f}\n'.format('recall:', R)
    message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', spe)
    message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', sen)
    message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
    #message += '{:>10}\t{:>10.4f}\n'.format('avg_time:', (end_time - start_time) / len(valLoader))
    message += '------------------------------------------------------\n'
    print(message)

    f.close()
    return acc

if __name__ == '__main__':
    #train(batch_size=8, lr=1e-3, epochs=200, dataroot='csv/feat.csv', label_type='train', dt='ASD')
    #train(batch_size=8, lr=1e-5, epochs=200, dataroot='csv/feat.csv', label_type='train', dt='A00')

    test(save_epoch='ours_best', dt='ASD')
    #test(save_epoch='ours_best', label_type='train', dt='ASD')

    # test(save_epoch='abide_best', dt='A00')
    # test(save_epoch='abide_best', label_type='train', dt='A00')

    #for i in range(0, 2):
    #    test(save_epoch=i)