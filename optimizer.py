import torch
import torch.optim as optim
def my_optimizer(parameters, epoch, lr):
    lr = learning_rate(lr, epoch)
    # if epoch <= 4:
    #     optimizer = torch.optim.Adam(parameters, lr=lr)
    # else:
    #     optimizer = torch.optim.SGD(parameters, lr=lr * 50)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    return optimizer

def learning_rate(lr, epoch):
    # if epoch < 4:
    #     factor = 1
    # elif epoch < 8:
    #     factor = 5
    # elif epoch < 12:
    #     factor = 20
    # else:
    #     factor = 50
    # return lr / factor
    return lr