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
from model.transformer import TransformerEncoderLayer
import torchvision.transforms as transforms

from model.feature_encoder import TEncoder


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # training params

    parser.add_argument('--bs', type=int, default=2,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=8,
                        help='number of workers for the dataloader')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--warm_epochs', default=0, type=int, help='warm up epochs  for learning rate')
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


class Net(nn.Module):
    def __init__(self, n_transformer_layer=3):
        super(Net, self).__init__()

        self.net = models.resnet50()

        self.tenc = nn.ModuleList()

        for i in range(n_transformer_layer):
            self.tenc.append(TransformerEncoderLayer(dims=2048))

        self.cls_token = torch.nn.Parameter(torch.randn([1, 1, 2048]) / np.sqrt(2048), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn(1, 50, 2048) / np.sqrt(2048), requires_grad=True)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  # bs x 2048 x 7 x 7

        bs = len(x)
        n_channels = x.shape[1]

        x = x.reshape(bs, n_channels, -1)
        x = x.permute(0, 2, 1)
        cls_token = self.cls_token.repeat(bs, 1, 1)

        x = torch.cat((x, cls_token), 1)

        pe = self.pos_emb
        for f in self.tenc:
            x = f(x, pe=pe)

        feat = x[:, -1]

        pred = self.net.fc(feat)
        return pred


def main(args):
    if 'aldb' in __file__:
        args.bs = 2
        args.nw = 1

    # copy this file to the log dir
    a = os.path.basename(__file__)

    file_name = a.split('.py')[0]

    exp_name = '%s/%s' % (file_name, args.trial)
    args.save_dir = os.path.join(args.save_dir, exp_name)

    # create the log folder
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    elif 'test' not in args.save_dir:
        ans = input('log dir exists, override? (yes=y)')

        print(ans, '<<<<<<<')

        if ans != 'y':
            exit()

    # creat logger
    logger = utils.Logger(args=args,
                          var_names=['Epoch', 'train_loss', 'train_acc', 'test_acc', 'best_acc', 'lr'],
                          format=['%02d', '%.4f', '%.4f', '%.3f', '%.3f', '%.6f'],
                          print_args=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    copyfile(os.path.join(dir_path, a), os.path.join(args.save_dir, a))

    # save args
    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    device = args.device

    acc_best = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ######################################################################################

    # Data loader for general contrastive learning =====================================================================
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val2')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=args.nw, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.nw, pin_memory=True)

    ######################################################################################

    net = Net()

    print('# of params in bbone    : %d' % utils.count_parameters(net))

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if args.chkpt != '':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.chkpt)
        net.load_state_dict(checkpoint['net'], 'cpu')
        start_epoch = args.startepoch

    net = nn.DataParallel(net)
    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)

    best_acc = 0

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if start_epoch > 0:
        scheduler.step(start_epoch)

    for epoch in range(start_epoch, args.epochs):

        lr = optimizer.param_groups[0]['lr']

        # break
        print("\n epoch: %02d, learning rate: %.4f" % (epoch, lr))
        t0 = time.time()

        train_acc, train_loss = train(
            epoch,
            net,
            optimizer,
            train_loader,
            device,
            args)

        print('>>>>>>>', args.save_dir)

        scheduler.step()

        # compute acc on nat examples
        test_acc, test_loss = validate(epoch, net, val_loader, device, args)

        print('%s: test acc: %.2f, best acc: %.2f' % (
            args.trial, test_acc, best_acc))

        state = {'net': net.module.state_dict(),
                 'acc': test_acc}
        torch.save(state, os.path.join(args.save_dir, 'model_last.pt'))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(state, os.path.join(args.save_dir, 'model_best.pt'))

        # print('test acc nat: %2.2f, best acc: %.2f' % (test_acc, acc_best))
        #
        logger.store(
            [epoch, train_loss, train_acc, test_acc, best_acc,
             optimizer.param_groups[0]['lr']],
            log=True)

        t = time.time() - t0
        remaining = (args.epochs - epoch) * t
        print("epoch time: %.1f, rt:%s" % (t, utils.format_time(remaining)))

        # scheduler.step()


def train(epoch, net, optimizer, trainloader, device, args):
    net.train()

    am_loss = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    prog_bar = tqdm(
        trainloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (img, target) in enumerate(prog_bar):

        # warm up learning rate
        if args.warm_epochs > 0 and epoch < args.warm_epochs:
            lr_ = args.lr * (batch_idx + 1 + epoch * len(prog_bar)) / (args.warm_epochs * len(prog_bar))
            optimizer.param_groups[0]['lr'] = lr_

        img, target = img.cuda(), target.cuda()

        optimizer.zero_grad()
        bs = len(img)

        logits = net(img)

        loss = criterion(logits, target)

        loss.backward()

        optimizer.step()

        pred_lbl = logits.argmax(1)

        correct += (pred_lbl == target).type(torch.float).sum()
        total += bs

        am_loss.update(loss.item())
        am_acc.update(correct / total)

        prog_bar.set_description(
            "E{}/{}, loss:{:2.3f}, loss_aux:{:2.3f}, acc:{:2.2f}, lr:{:2.8f}".format(
                epoch, args.epochs, am_loss.avg, 0, correct * 100 / total, optimizer.param_groups[0]['lr']))
        # print(net.module.cls_token[0, 0, 0])
    prog_bar.close()

    acc = correct * 100 / total

    return acc, am_loss.avg


def validate(epoch, net, valloader, device, args):
    net.eval()

    am_loss = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    prog_bar = tqdm(
        valloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(prog_bar):
            img, target = img.cuda(), target.cuda()

            bs = len(img)

            logits = net(img)

            loss = criterion(logits, target)
            pred_lbl = logits.argmax(1)
            correct += (pred_lbl == target).type(torch.float).sum()
            total += bs

            am_loss.update(loss.item())
            am_acc.update(correct / total)

            prog_bar.set_description(
                "eval: E{}/{}, loss:{:2.3f}, loss_aux:{:2.3f}, acc:{:2.2f}".format(
                    epoch, args.epochs, am_loss.avg, 0, correct * 100 / total))
    prog_bar.close()

    acc = correct * 100 / total

    return acc, am_loss.avg


if __name__ == "__main__":
    args = get_params(train=True)

    args.gpu_devices = [int(id) for id in args.gpu_id.split(',')]

    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]

    print('gpu devices to use: ', args.gpu_devices)

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    main(args)
