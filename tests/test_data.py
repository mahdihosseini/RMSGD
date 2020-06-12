from pathlib import Path

from torch.utils.data import DataLoader

from adas.data import get_data
from support import fail_if


def test_cifar10():
    train_loader, test_loader = get_data(root=Path('/tmp'),
                                         dataset='CIFAR10', mini_batch_size=1)
    fail_if(not isinstance(train_loader, DataLoader))
    fail_if(not isinstance(test_loader, DataLoader))
    fail_if(not Path('/tmp/cifar-10-batches-py').is_dir())


def test_cifar100():
    train_loader, test_loader = get_data(root=Path('/tmp'),
                                         dataset='CIFAR100', mini_batch_size=1)
    fail_if(not isinstance(train_loader, DataLoader))
    fail_if(not isinstance(test_loader, DataLoader))
    fail_if(not Path('/tmp/cifar-100-python').is_dir())


def test_imagenet():
    train_loader, test_loader = get_data(root=Path('/tmp'),
                                         dataset='ImageNet', mini_batch_size=1)
    fail_if(not isinstance(train_loader, DataLoader))
    fail_if(not isinstance(test_loader, DataLoader))
    # fail_if(not Path('/tmp/cifar-100-python').is_dir())
