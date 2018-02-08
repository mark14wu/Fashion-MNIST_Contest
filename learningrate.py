def learning_rate(lr, epoch):
    if epoch < 3:
        factor = 1
    elif epoch < 33:
        factor = 10
    # elif epoch < 9:
    #     factor = 100
    else:
        factor = 100
    return lr / factor