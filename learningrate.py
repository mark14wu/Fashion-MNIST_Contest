def learning_rate(lr, epoch):
    if epoch < 3:
        factor = 1
    elif epoch < 4:
        factor = 10
    else:
        factor = 20
    return lr / factor