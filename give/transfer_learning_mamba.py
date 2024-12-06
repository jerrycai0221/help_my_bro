import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import os
import argparse

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True, help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')  # 增加训练轮数
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')  # 使用较小学习率
parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')  # 减小 weight decay
parser.add_argument('--hidden', type=int, default=32, help='Dimension of representations')
parser.add_argument('--layer', type=int, default=4, help='Num of layers')  # 增加层数
parser.add_argument('--task', type=str, default='SOH', help='RUL or SOH')
parser.add_argument('--case', type=str, default='B', help='A or B')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()


# 设置随机种子
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


set_seed(args.seed, args.cuda)


# 评估指标
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))


# 读取数据
def ReadData(path, xlsx, task):
    f = os.path.join(path, xlsx)
    data = pd.read_excel(f)  # 修改为 pd.read_excel 以读取 xlsx 文件
    tf = len(data)
    y = data[task]
    y = y.values
    if args.task == 'RUL': y = y / tf
    x = data.drop(['RUL', 'SOH'], axis=1).values
    x = scale(x)
    return x, y


# 网络结构
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # 设置 d_model 为输入数据的特征维度（如果数据是 7 特征，d_model 也设为 7）
        self.config = MambaConfig(d_model=in_dim, n_layers=args.layer)
        self.mamba = Mamba(self.config)

        # 新增全连接层（fc层），用于将 Mamba 输出的特征转化为最终的预测
        self.fc = nn.Linear(self.config.d_model, out_dim)  # 动态设置fc层的输入维度

        # 添加 dropout 层来减少过拟合
        self.dropout = nn.Dropout(p=0.3)  # 减少 dropout 比例

        # 添加 Sigmoid 层，用于输出预测结果
        self.sigmoid = nn.Sigmoid()

        # 仅解冻最后的 fc 层，其他层冻结
        for param in self.mamba.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 通过冻结的 Mamba 模型进行特征提取
        x = self.mamba(x)

        # 通过训练的 fc 层进行任务输出
        x = self.fc(x)

        # 添加 dropout 层
        x = self.dropout(x)

        # 激活函数，如果是分类任务，Sigmoid 可以改为 softmax 或其他
        x = self.sigmoid(x)

        return x.flatten()  # 返回展平的输出，以便后续处理


# 训练过程
def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]), 1)  # 使用迁移学习模型
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)  # 训练所有参数

    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()

    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()

    # 使用学习率调度器来动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)  # 使用 MSE 损失
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()  # 每个周期调整一次学习率

        if e % 10 == 0 and e != 0:
            print(f'Epoch {e} | Loss: {loss.item()}')

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat


# 主程序入口
path = r'D:\mamba\MambaLithium-main\data\Case' + args.case
if args.case == 'A':
    xt1, yt1 = ReadData(path, 'cx2_36_update.xlsx', args.task)  # 使用 xlsx 文件名
    xt2, yt2 = ReadData(path, 'cx2_37_update.xlsx', args.task)
    trainX = np.vstack((xt1, xt2))
    trainy = np.hstack((yt1, yt2))
    testX, testy = ReadData(path, 'cx2_38_update.xlsx', args.task)
else:
    xt1, yt1 = ReadData(path, 'CS2_35_final.xlsx', args.task)
    xt2, yt2 = ReadData(path, 'CS2_36_final.xlsx', args.task)
    xt3, yt3 = ReadData(path, 'CS2_37_final.xlsx', args.task)
    trainX = np.vstack((xt1, xt2, xt3))
    trainy = np.hstack((yt1, yt2, yt3))
    testX, testy = ReadData(path, 'CS2_38_final.xlsx', args.task)

# 使用迁移学习进行预测
predictions_transfer = PredictWithData(trainX, trainy, testX)

# 打印评估结果
print('MSE RMSE MAE R2')
evaluation_metric(testy, predictions_transfer)

# 绘制预测与真实值对比图
plt.figure()
plt.plot(testy, label='True')
plt.plot(predictions_transfer, label='Estimation (Transfer Learning)')
plt.title(args.task + ' Estimation with Transfer Learning')
plt.xlabel('Cycle')
plt.ylabel(args.task + ' value')
plt.legend()
plt.show()
