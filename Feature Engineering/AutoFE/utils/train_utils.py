from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
from torch.utils import data as Data
import numpy as np
from torch.utils.tensorboard import SummaryWriter

log_dir: str = 'log/'


def train_test_split(x, y, random_state):
    return train_test_split(x, y, random_state=random_state)


def linereg(x, y):
    return LinearRegression().fit(x, y)


def torch_linear(x_train, y_train, x_test, y_test, log_name, batch_size=None, optimizer=None, net=None):
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    num_inputs = x_train.shape[1]
    dataset = Data.TensorDataset(x_train, y_train)
    if batch_size is None:
        batch_size = 256
    data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if net is None:
        net = nn.Sequential(
            nn.Linear(num_inputs, 1),
            nn.ReLU()
        )
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.constant_(net[0].bias, val=0)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epoch = 16
    loss = nn.MSELoss()
    writer_train = SummaryWriter(log_dir+log_name+'/train')
    writer_test = SummaryWriter(log_dir+log_name+'/test')
    for epoch in range(num_epoch):
        for x, y in data_iter:
            output = net(x)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))
        writer_train.add_scalar("loss", l.item(), epoch)
        get_metrics(net, writer_train, x_train, y_train, epoch)
        get_metrics(net, writer_test, x_test, y_test, epoch)
    writer_test.close()
    writer_train.close()
    return net


def get_metrics(net, writer, x_test, y_test, epoch):
    from sklearn import metrics
    out = net(torch.tensor(np.array(x_test), dtype=torch.float32)).detach().numpy()
    writer.add_scalar("MSE:", metrics.mean_squared_error(y_test, out), epoch)
    writer.add_scalar("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, out)), epoch)
    writer.add_scalar("MAE", metrics.median_absolute_error(y_test, out), epoch)
    writer.add_scalar("R2", metrics.r2_score(y_test, out), epoch)
    # print("MSE:", metrics.mean_squared_error(y_test, out))
    # # 用scikit-learn计算RMSE
    # print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, out)))
    # # print("MAE:",metrics.mean_squared_error(y_test, out))
    # print("MAE", metrics.median_absolute_error(y_test, out))
    # print("R2", metrics.r2_score(y_test, out))


def linearNet(x_train, y_train, batch_size=None, optimizer=None, net=None):
    num_inputs = x_train.shape[1]
    dataset = Data.TensorDataset(x_train, y_train)
    if batch_size is None:
        batch_size = 256
    data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if net is None:
        net = nn.Sequential(
            nn.Linear(num_inputs, 1),
            nn.ReLU()
        )
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.constant_(net[0].bias, val=0)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epoch = 16
    loss = nn.MSELoss()

    for epoch in range(num_epoch):
        for x, y in data_iter:
            output = net(x)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))
    return net


def mean_squared_error(data_copy, s, y):
    from sklearn import metrics
    x = data_copy[s].__array__()
    x = torch.tensor(np.array(x), dtype=torch.float32)
    x = x.reshape([-1, 1])
    net = linearNet(x, y)

    out = net(torch.tensor(np.array(x), dtype=torch.float32)).detach().numpy()
    y_ = y.detach().numpy()
    return np.sqrt(metrics.mean_squared_error(y_, out))

