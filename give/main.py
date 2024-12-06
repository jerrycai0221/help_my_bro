import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--task', type=str, default='SOH',
                    help='RUL or SOH')
parser.add_argument('--case', type=str, default='A',
                    help='A or B')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim,args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden,out_dim),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.mamba(x)
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]),1)
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.l1_loss(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

def ReadData(path,csv,task):
    f = os.path.join(path,csv)
    data = pd.read_csv(f)
    tf = len(data)
    y = data[task]
    y = y.values
    if args.task == 'RUL': y = y/tf
    x = data.drop(['RUL','SOH'],axis=1).values
    x = scale(x)
    return x,y

path = './data/Case'+args.case
if args.case == 'A':
    xt1, yt1 = ReadData(path,'91.csv',args.task)
    xt2, yt2 = ReadData(path,'100.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX,testy = ReadData(path,'124.csv',args.task)
else:
    xt1, yt1 = ReadData(path,'101.csv',args.task)
    xt2, yt2 = ReadData(path,'108.csv',args.task)
    xt3, yt3 = ReadData(path,'120.csv',args.task)
    trainX = np.vstack((xt1,xt2,xt3))
    trainy = np.hstack((yt1,yt2,yt3))
    testX,testy = ReadData(path,'116.csv',args.task)

predictions = PredictWithData(trainX, trainy, testX)
tf = len(testy)
if args.task == 'RUL':
    testy = tf*testy
    predictions = tf*predictions

print('MSE RMSE MAE R2')
evaluation_metric(testy, predictions)
plt.figure()
plt.plot(testy, label='True')
plt.plot(predictions, label='Estimation')
plt.title(args.task+' Estimation')
plt.xlabel('Cycle')
plt.ylabel(args.task+' value')
plt.legend()
plt.show()