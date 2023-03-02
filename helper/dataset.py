import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

class DatasetWithIndex(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return (img, label, index)

    def __len__(self):
        return len(self.dataset)

def get_dataset(args, include_index=False, train_shuffle=True):
    if args.dataset == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        if include_index:
            trainset = DatasetWithIndex(trainset)


        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=train_shuffle, num_workers=4)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=200, shuffle=False, num_workers=4)
        return trainloader, testloader
    elif args.dataset == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        if include_index:
            trainset = DatasetWithIndex(trainset)


        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=train_shuffle, num_workers=4)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=200, shuffle=False, num_workers=4)
        return trainloader, testloader
    elif args.dataset == 'svhn':

        trainloader = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(
                    root='./data', split='train', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                ),
                batch_size=args.bs, shuffle=True, num_workers=4)

        testloader = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(
                    root='./data', split='test', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
                batch_size=200, shuffle=False, num_workers=4)
        return trainloader, testloader

    else:
        raise NotImplementedError
