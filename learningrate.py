def learning_rate(lr, epoch):
    if epoch < 3:
        factor = 1
    elif epoch < 6:
        factor = 10
    elif epoch < 9:
        factor = 100
    else:
        factor = 1000
    return lr / factor