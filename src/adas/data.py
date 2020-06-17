"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

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

import torchvision.transforms as transforms

import torchvision
import torch


def get_data(root: Path, dataset: str, mini_batch_size: int):
    train_loader = None
    test_loader = None
    if dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.ImageNet(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)

        testset = torchvision.datasets.ImageNet(
            root=str(root), train=False, download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'COCO':
        ...
    return train_loader, test_loader
