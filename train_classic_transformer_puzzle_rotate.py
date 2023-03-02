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

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, rotate=False):
        self.dataset = dataset
        self.rotate = rotate
        if self.rotate:
            print('augmentation includes rotating....')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # print(img.shape)

        if self.rotate:
            p = np.random.rand(1)

            if p > 0.5:
                k = np.random.randint(1, 4)
                # print(k, '<<<<<<<<<')
                img = torch.rot90(img, dims=[1, 2], k=k)
            # utils.plot_tensor([img[None, ...], img_shuffled[None, ...]])

        # x = torch.randn(1, 500, 500, 500)  # batch, c, h, w
        kc, kh, kw = 3, 32, 32  # kernel size
        dc, dh, dw = 3, 32, 32  # stride
        patches = img.unsqueeze(0).unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)

        unfold_shape = patches.size()
        patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
        # print(patches.shape)
        # exit()

        shuffle = torch.randperm(49)
        patches_shuffled = patches[:, shuffle]

        # Reshape back
        patches_orig = patches_shuffled.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        img_shuffled = patches_orig.view(1, output_c, output_h, output_w).squeeze(0)



        return img, img_shuffled, shuffle, target


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--net', type=str, default='resnet18',
                        help='device for training the model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--rotate', type=bool, default=False,
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # training params

    parser.add_argument('--bs', type=int, default=2,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=8,
                        help='number of workers for the dataloader')
    parser.add_argument('--nt', type=int, default=3,
                        help='number of transformer layers')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

    parser.add_argument('--a', default=0.4, type=float, help='coef for classification loss')
    parser.add_argument('--b', default=0.1, type=float, help='coef for suppression loss')

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


class Net(nn.Module):
    def __init__(self, args=None):
        super(Net, self).__init__()

        if args.net == 'resnet18':
            self.net = models.resnet18()
        else:
            self.net = models.resnet50()

        ndim = self.net.fc.weight.shape[1]

        self.tenc = nn.ModuleList()

        for i in range(args.nt):
            self.tenc.append(TransformerEncoderLayer(dims=ndim))

        self.cls_token = torch.nn.Parameter(torch.randn([1, 1, ndim]) / np.sqrt(ndim), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn(1, 50, ndim) / np.sqrt(ndim), requires_grad=True)
    def forward(self, x, x_s=None, shuffle_idx=None):

        if x_s is not None:
            x_tot = torch.cat((x, x_s), 0)

        else:
            x_tot = x

        x = self.net.conv1(x_tot)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  # bs x 2048 x 7 x 7

        bs = len(x) // 2
        n_channels = x.shape[1]

        pred_a = self.encode(x, self.pos_emb)

        if shuffle_idx is None:
            return pred_a

        pred_normalimg_normalidx, pred_shuffleimg_normalidx = pred_a[:bs], pred_a[bs:]

        # shuffle pos embedding ####################################################################################
        pe = self.pos_emb[0, :-1].view(1, 49, -1)

        l = []

        for i_ in range(bs):
            l.append(pe[:, shuffle_idx[i_]])
        pe_shuffled = torch.cat(l, 0)
        pe_shuffled = torch.cat((pe_shuffled, self.pos_emb[:, 49:].repeat(bs, 1, 1)), 1)

        pred_shuffleimg_shuffleidx = self.encode(x[bs:], pe_shuffled)
        pred_normalimg_shuffleidx = self.encode(x[:bs], pe_shuffled)

        # error = torch.abs(pred_normalimg_normalidx-pred_shuffleimg_shuffleidx).sum()
        # print(error)
        # error = torch.abs(pred_normalimg_normalidx-pred_shuffleimg_normalidx).sum()
        # print(error)
        # exit()

        return pred_normalimg_normalidx, pred_shuffleimg_shuffleidx, pred_normalimg_shuffleidx, pred_shuffleimg_normalidx

    def encode(self, x, pe):

        bs = len(x)
        n_channels = x.shape[1]
        x = x.reshape(bs, n_channels, -1)
        x = x.permute(0, 2, 1)

        cls_token = self.cls_token.repeat(bs, 1, 1)

        x = torch.cat((x, cls_token), 1)

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
                          var_names=['Epoch', 'l_nn', 'l_ss', 'l_ns', 'l_sn', 'acc_nn', 'acc_ss', 'acc_ns', 'acc_sn',
                                     'test_acc', 'best_acc', 'lr'],
                          format=['%02d', '%.4f', '%.4f', '%.4f', '%.4f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f',
                                  '%.3f', '%.6f'],
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

    train_dataset = CustomDataset(train_dataset, rotate=args.rotate)

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
        num_workers=args.nw, pin_memory=True, drop_last=True)

    val_dataset = CustomDataset(val_dataset, rotate=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.nw, pin_memory=True)

    ######################################################################################

    net = Net(args=args)

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

        l_nn, l_ss, l_ns, l_sn, a_nn, a_ss, a_ns, a_sn = train(
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
            [epoch, l_nn, l_ss, l_ns, l_sn, a_nn, a_ss, a_ns, a_sn, test_acc, best_acc,
             optimizer.param_groups[0]['lr']],
            log=True)

        t = time.time() - t0
        remaining = (args.epochs - epoch) * t
        print("epoch time: %.1f, rt:%s" % (t, utils.format_time(remaining)))

        # scheduler.step()


def kldiv2(x, y):
    x = F.log_softmax(x, 1)
    return nn.KLDivLoss(reduction='none')(x, y).sum(1)


def comp_acc(pred, target):
    pred_lbl = pred.argmax(1)
    acc = (pred_lbl == target).type(torch.float).mean()
    return acc * 100


def train(epoch, net, optimizer, trainloader, device, args):
    net.train()

    am_loss_nn = utils.AverageMeter()
    am_loss_ss = utils.AverageMeter()
    am_loss_ns = utils.AverageMeter()
    am_loss_sn = utils.AverageMeter()
    am_acc_nn = utils.AverageMeter()
    am_acc_ss = utils.AverageMeter()
    am_acc_ns = utils.AverageMeter()
    am_acc_sn = utils.AverageMeter()

    prog_bar = tqdm(
        trainloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (img, img_shuffle, shuffle_idx, target) in enumerate(prog_bar):
        # if batch_idx>0:
        #     break

        # warm up learning rate
        if args.warm_epochs > 0 and epoch < args.warm_epochs:
            lr_ = args.lr * (batch_idx + 1 + epoch * len(prog_bar)) / (args.warm_epochs * len(prog_bar))
            optimizer.param_groups[0]['lr'] = lr_

        img, img_shuffle, shuffle_idx, target = img.cuda(), img_shuffle.cuda(), shuffle_idx.cuda(), target.cuda()

        optimizer.zero_grad()
        bs = len(img)

        pred_nimg_nidx, pred_simg_sidx, pred_nimg_sidx, pred_simg_nidx = net(img, img_shuffle, shuffle_idx)

        target_uniform = torch.ones(bs, 1000).cuda() / 1000

        loss_nn = criterion(pred_nimg_nidx, target)
        loss_ss = criterion(pred_simg_sidx, target)

        loss_ns = -torch.mean(torch.sum(F.log_softmax(pred_nimg_sidx, dim=1) * target_uniform, dim=1))
        loss_sn = -torch.mean(torch.sum(F.log_softmax(pred_simg_nidx, dim=1) * target_uniform, dim=1))

        # loss_ns = kldiv2(pred_nimg_sidx, target_uniform).mean()
        # loss_sn = kldiv2(pred_simg_nidx, target_uniform).mean()

        loss = loss_nn * args.a + loss_ss * args.a + loss_ns * args.b + loss_sn * args.b

        loss.backward()

        optimizer.step()

        am_acc_nn.update(comp_acc(pred_nimg_nidx, target))
        am_acc_ss.update(comp_acc(pred_simg_sidx, target))
        am_acc_ns.update(comp_acc(pred_nimg_sidx, target))
        am_acc_sn.update(comp_acc(pred_simg_nidx, target))

        am_loss_nn.update(loss_nn.item())
        am_loss_ss.update(loss_ss.item())
        am_loss_sn.update(loss_sn.item())
        am_loss_ns.update(loss_ns.item())

        prog_bar.set_description(
            "E{}/{}, l_nn:{:2.5f}, l_ss:{:2.5f}, l_ns:{:2.5f}, l_sn:{:2.5f}, acc_nn:{:2.2f}, acc_ss:{:2.2f}, acc_ns:{:2.2f}, acc_sn:{:2.2f}, lr:{:2.8f}".format(
                epoch, args.epochs, am_loss_nn.avg, am_loss_ss.avg, am_loss_ns.avg, am_loss_sn.avg, am_acc_nn.avg,
                am_acc_ss.avg, am_acc_ns.avg, am_acc_sn.avg, optimizer.param_groups[0]['lr']))
        # print(net.module.cls_token[0, 0, 0])
    prog_bar.close()

    return am_loss_nn.avg, am_loss_ss.avg, am_loss_ns.avg, am_loss_sn.avg, am_acc_nn.avg, am_acc_ss.avg, am_acc_ns.avg, am_acc_sn.avg


def validate(epoch, net, valloader, device, args):
    net.eval()

    am_loss = utils.AverageMeter()
    am_acc = utils.AverageMeter()

    am_loss_nn = utils.AverageMeter()
    am_loss_ss = utils.AverageMeter()
    am_loss_ns = utils.AverageMeter()
    am_loss_sn = utils.AverageMeter()
    am_acc_nn = utils.AverageMeter()
    am_acc_ss = utils.AverageMeter()
    am_acc_ns = utils.AverageMeter()
    am_acc_sn = utils.AverageMeter()

    prog_bar = tqdm(
        valloader,
        ascii=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
    )

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (img, img_shuffle, shuffle_idx, target) in enumerate(prog_bar):
            img, img_shuffle, shuffle_idx, target = img.cuda(), img_shuffle.cuda(), shuffle_idx.cuda(), target.cuda()

            bs = len(img)

            pred_nimg_nidx, pred_simg_sidx, pred_nimg_sidx, pred_simg_nidx = net(img, img_shuffle, shuffle_idx)

            target_uniform = torch.ones(bs, 1000).cuda() / 1000

            loss_nn = criterion(pred_nimg_nidx, target)
            loss_ss = criterion(pred_simg_sidx, target)

            loss_ns = -torch.mean(torch.sum(F.log_softmax(pred_nimg_sidx, dim=1) * target_uniform, dim=1))
            loss_sn = -torch.mean(torch.sum(F.log_softmax(pred_simg_nidx, dim=1) * target_uniform, dim=1))

            am_acc.update(comp_acc(pred_nimg_nidx, target))
            am_acc_ss.update(comp_acc(pred_simg_sidx, target))
            am_acc_ns.update(comp_acc(pred_nimg_sidx, target))
            am_acc_sn.update(comp_acc(pred_simg_nidx, target))

            am_loss.update(loss_nn.item())
            am_loss_ss.update(loss_ss.item())
            am_loss_sn.update(loss_sn.item())
            am_loss_ns.update(loss_ns.item())

            # print(pred_nimg_nidx.shape)
            # print(target.shape)
            # print(img.shape)


            correct += (pred_nimg_nidx.argmax(1) == target).type(torch.float).sum().item()
            total += bs


            prog_bar.set_description(
                "eval: E{}/{}, l_nn:{:2.5f}, l_ss:{:2.5f}, l_ns:{:2.5f}, l_sn:{:2.5f}, acc_nn:{:2.2f}, acc_ss:{:2.2f}, acc_ns:{:2.2f}, acc_sn:{:2.2f}".format(
                    epoch, args.epochs, am_loss.avg, am_loss_ss.avg, am_loss_ns.avg, am_loss_sn.avg, am_acc.avg,
                    am_acc_ss.avg, am_acc_ns.avg, am_acc_sn.avg))
            # print(net.module.cls_token[0, 0, 0])
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
