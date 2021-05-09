"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path
import sys

import torchvision.transforms as transforms

import torchvision
import numpy as np
import torch

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .datasets import ImageNet, TinyImageNet, MHIST
else:
    from datasets import ImageNet, TinyImageNet, MHIST
# from .folder2lmdb import ImageFolderLMDB


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    @author: uoguelph-mlrg
      (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        if n_holes < 0 or length < 0:
            raise ValueError("Must set n_holes or length args for cutout")
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of
            it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_data(
        name: str, root: Path,
        mini_batch_size: int,
        num_workers: int,
        cutout: bool = False,
        n_holes: int = -1,
        length: int = -1,
        dist: bool = False) -> None:
    if name == 'MHIST':
        num_classes = 2
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [188.14, 165.39, 192.69]], std=[
                    x / 255.0 for x in [50.30, 62.13, 43.42]]),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [188.14, 165.39, 192.69]], std=[
                    x / 255.0 for x in [50.30, 62.13, 43.42]]),
        ])
        trainset = MHIST(
            root=str(root), split='train',
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)
        testset = MHIST(
            root=str(root), split='test',
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'CIFAR10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)

        testset = torchvision.datasets.CIFAR10(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'CIFAR100':
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)
        testset = torchvision.datasets.CIFAR100(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'CIFAR10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)

        testset = torchvision.datasets.CIFAR10(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'ImageNet':
        num_classes = 1000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        trainset = ImageNet(
            root=str(root), split='train', download=None,
            transform=transform_train)
        # trainset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'train'),
        #     transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        # trainset = ImageFolderLMDB(str(root / 'train.lmdb'),
        #                            transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True, sampler=train_sampler)

        # testset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'val'),
        #     transform=transform_test)
        testset = ImageNet(
            root=str(root), split='val', download=None,
            transform=transform_test)
        # testset = ImageFolderLMDB(str(root / 'val.lmdb'),
        #                           transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'TinyImageNet':
        num_classes = 200
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        if cutout:
            transform_train.transforms.append(
                Cutout(n_holes=n_holes, length=length))

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        trainset = TinyImageNet(
            root=str(root), split='train', download=False,
            transform=transform_train)
        # trainset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'train'),
        #     transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        # trainset = ImageFolderLMDB(str(root / 'train.lmdb'),
        #                            transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True, sampler=train_sampler)

        # testset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'val'),
        #     transform=transform_test)
        testset = TinyImageNet(
            root=str(root), split='val', download=False,
            transform=transform_test)
        # testset = ImageFolderLMDB(str(root / 'val.lmdb'),
        #                           transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    return train_loader, train_sampler, test_loader, num_classes
