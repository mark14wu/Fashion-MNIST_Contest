import torch.optim as optim
def my_optimizer(parameters, epoch, lr):
    lr = learning_rate(lr, epoch)
    if epoch <= 4:
        optimizer = torch.optim.Adam(parameters, lr)
    else:
        optimizer = torch.optim.SGD(parameters, lr * 50)
    return optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate(args.lr, e))

def learning_rate(lr, epoch):
    if epoch < 4:
        factor = 1
    elif epoch < 8:
        factor = 2
    else epoch < 12:
        factor = 3
    else:
        factor = 4
    return lr / factor