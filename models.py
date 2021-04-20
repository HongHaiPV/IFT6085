import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data

# from tqdm import tqdm
tqdm = lambda x: x

class FeedForward(nn.Module):
    def __init__(self, depth, width):
        super(FeedForward, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return x


def train_network(train_loader, test_loader, depth, width, criterion, epochs, init_lr, decay, loss_lr):
    
    model = FeedForward(depth, width).cuda()
    net_optim = Nero(model.parameters(), lr=init_lr)
    loss_optim = torch.optim.SGD(criterion.parameters(), lr=loss_lr)

    lr_lambda = lambda x: decay**x
    net_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(net_optim, lr_lambda)
    loss_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(loss_optim, lr_lambda)

    model.train()
    
    train_acc_list = []

    for epoch in range(epochs):

        correct = 0
        total = 0

        for data, target in tqdm(train_loader):
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)

            feats = model(data)
            loss, y_pred = criterion(feats, target)

            correct += (target.float() == y_pred).sum().item()
            total += target.shape[0]

            net_optim.zero_grad()
            loss_optim.zero_grad()

            loss.backward()

            net_optim.step()
            loss_optim.step()

        net_lr_scheduler.step()
        loss_lr_scheduler.step()
        train_acc_list.append(correct/total)

    model.eval()
    correct = 0
    total = 0

    for data, target in tqdm(test_loader):
        data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target)
        
        feats = model(data)
        _, y_pred =  criterion(feats, target).squeeze()
        correct += (target.float() == y_pred).sum().item()
        total += target.shape[0]

    test_acc = correct/total
  
    return train_acc_list, test_acc, model