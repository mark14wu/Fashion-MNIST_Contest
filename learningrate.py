def learning_rate(args.lr, epoch):
    if epoch < 3:
        factor = 1
    elif epoch < 6:
        factor = 2
    elif epoch < 9:
        factor = 3
    else:
        factor = 5
    return args.lr / factor