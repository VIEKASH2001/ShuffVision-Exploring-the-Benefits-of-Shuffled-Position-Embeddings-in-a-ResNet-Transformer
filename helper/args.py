import argparse
import os


def get_params(train=False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # general params
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset can be cifar10, svhn')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training the model')
    parser.add_argument('--gpu_id', type=str, default='0,1,2',
                        help='device for training the model')
    parser.add_argument('--trial', type=str, default='test',
                        help='name for the experiment')
    parser.add_argument('--chkpt', type=str, default='',
                        help='checkpoint for resuming the training')

    # dense contrastive params


    parser.add_argument('--n_neg', type=int, default=8,
                        help='number of negative local samples')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='feature dim for the output of the backbone')
    parser.add_argument('--neg_m', type=int, default=3, help='number of random negative samples')
    parser.add_argument('--lam_cont', default=1, type=float, help='global contrastive loss coef')
    parser.add_argument('--lam_dense', default=1, type=float, help='local contrastive loss coef')
    parser.add_argument('--m', default=65536, type=int, help='queue size for moco')
    parser.add_argument('--moco_momentum', default=0.999, type=float, help='queue size for moco')
    parser.add_argument('--temp', default=0.07, type=float, help='temprature for contrastive learning')

    # training params

    parser.add_argument('--bs', type=int, default=128,
                        help='batchsize')
    parser.add_argument('--nw', type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--save_dir', type=str, default='./logs',
                        help='directory to log and save checkpoint')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_policy', default='cosine', type=str, choices=['cosine', 'step'], help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160',
                        help='where to decay lr, can be a list')
    parser.add_argument('--datadir', type=str, default='/home/aldb/dataset/ILSVRC/Data/CLS-LOC',
                        help='directory of the data')

    args = parser.parse_args()

    if train and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args
