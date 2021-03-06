import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from fashionmnist import FashionMNIST
from torchvision import datasets
import model
import utils
import time
import argparse
import os
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
import numpy as np
# from learningrate import learning_rate
from optimizer import my_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='resnet18', help="model")
parser.add_argument("--patience", type=int, default=200, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("--nworkers", type=int, default=36, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--data", type=str, default='fashion', help="mnist or fashion")
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('Training on cuda: {}'.format(cuda))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup tensorboard folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
	run += 1
	current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

# Tensorboard viz. tensorboard --logdir {yourlogdir}. Requires tensorflow.
from tensorboard_logger import configure, log_value
configure(current_dir, flush_secs=5)

# Define transforms.
# normalize = transforms.Normalize((0.1307,), (0.3081,)
train_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        utils.RandomRotation(),
                        utils.RandomTranslation(),
                        utils.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]
                        )
val_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])

# Create dataloaders. Use pin memory if cuda.
kwargs = {'pin_memory': True} if cuda else {}
if(args.data == 'mnist'):
    trainset = datasets.MNIST('data-mnist', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers, **kwargs)
    valset = datasets.MNIST('data-mnist', train=False, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.nworkers, **kwargs)
else:
    trainset = FashionMNIST('data', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.nworkers, **kwargs)
    valset = FashionMNIST('data', train=False, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.nworkers, **kwargs)

def train(net, loader, criterion, optimizer):
    net.train()
    running_loss = 0
    running_accuracy = 0
    output_list = []
    y_list = []
    for i, (X,y) in enumerate(loader):
        y_list += list(y)
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        output_list += list(pred)
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
    my_f1_score = f1_score(y_list, output_list, average='weighted')
    return running_loss/len(loader), running_accuracy/len(loader.dataset), my_f1_score

def validate(net, loader, criterion):
    net.eval()
    running_loss = 0
    running_accuracy = 0
    y_list=[]
    output_list=[]
    for i, (X,y) in enumerate(loader):
        y_list+=list(y)
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        running_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        output_list+=list(pred)
        running_accuracy += pred.eq(y.data.view_as(pred)).cpu().sum()
    my_f1_score = f1_score(y_list, output_list, average='weighted')
    return running_loss/len(loader), running_accuracy/len(loader.dataset),my_f1_score


if __name__ == '__main__':
    net = model.__dict__[args.model]()
    print(net)
    criterion = torch.nn.CrossEntropyLoss()
    writer=SummaryWriter('mylog/')
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()
        # net = torch.nn.DataParallel(net).cuda()
        # criterion = torch.nn.DataParallel(criterion).cuda()
    # early stopping parameters
    patience = args.patience
    best_loss = 1e4

    # Print model to logfile
    print(net, file=logfile)

    for e in range(args.nepochs):
        # Change optimizer for finetuning
        optimizer = my_optimizer(net.parameters(), e, args.lr)
        start = time.time()
        train_loss, train_acc, train_f1 = train(net, train_loader,
            criterion, optimizer)
        val_loss, val_acc, val_f1 = validate(net, val_loader, criterion)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train f1score: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f},val f1score: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format( e, train_loss,train_f1, train_acc, val_loss,
                val_f1,val_acc, end-start)
        print(stats)
        print(stats, file=logfile)
        writer.add_scalar('train_loss', train_loss, e)
        writer.add_scalar('val_loss', val_loss, e)
        writer.add_scalar('train_acc', train_acc, e)
        writer.add_scalar('val_acc', val_acc, e)
        writer.add_scalar('train_f1', val_f1, e)
        writer.add_scalar('val_f1', val_f1, e)

        log_value('train_loss', train_loss, e)
        log_value('val_loss', val_loss, e)
        log_value('train_acc', train_acc, e)
        log_value('val_acc', val_acc, e)
        log_value('train_f1', val_f1, e)
        log_value('val_f1', val_f1, e)

        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break
