import copy
import os
import argparse
import os
import time
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
from helper import utils
from shutil import copyfile
import pickle
from torch.utils.data import Dataset

# from models.resnet import resnet50, resnet18
import torchvision.models as models
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from train_classic_transformer_puzzle_rotate import Net


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--net', type=str, default='resnet18',
                        help='device for training the model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # training params

    parser.add_argument('--bs', type=int, default=1,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=1,
                        help='number of workers for the dataloader')
    parser.add_argument('--nt', type=int, default=3,
                        help='number of workers for the dataloader')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--warm_epochs', default=5, type=int, help='warm up epochs  for learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
    parser.add_argument('--epochs', default=90, type=int, help='training epochs')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60',
                        help='where to decay lr, can be a list')
    parser.add_argument('--datadir', type=str, default='/home/aldb/dataset/ILSVRC/Data/CLS-LOC',
                        help='directory of the data')

    args = parser.parse_args()

    if train and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


args = get_params(train=True)

traindir = os.path.join(args.datadir, 'train')
valdir = os.path.join(args.datadir, 'val2')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

import matplotlib
print(matplotlib.get_backend())
exit()

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.bs, shuffle=False,
    num_workers=args.nw, pin_memory=True)

net1 = Net(args)
net2 = Net(args)

net1.load_state_dict(
    torch.load('/home/aldb/mount_scratch_2/chkpt/spatial_org/train_classic_transformer/res18_0.1/model_best.pt')['net'])

net1 = net1.cuda()
net1.eval()

net2.load_state_dict(
    torch.load('/home/aldb/mount_scratch_2/chkpt/spatial_org/train_classic_transformer_puzzle_rotate/norotate_0.4_0.4_0.1_0.1/model_best.pt')['net'])

net2 = net2.cuda()
net2.eval()


total = 0
correct1 = 0
correct2 = 0

for i, (img, target) in enumerate(val_loader):
    img, target = img.cuda(), target.cuda()

    bs = len(img)
    pred1 = net1(img)
    pred_lbl1 = pred1.argmax(1)
    c1 = (pred_lbl1 == target).type(torch.float).sum().item()

    pred2 = net2(img)
    pred_lbl2 = pred2.argmax(1)
    c2 = (pred_lbl2 == target).type(torch.float).sum().item()

    if c1!=c2:
        print(c1, c2)
        utils.plot_tensor([img])


    total += bs

    correct1 += c1
    correct2 += c2

    print(i, correct1/total, correct2/total)





