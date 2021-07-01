# code adapted from https://github.com/pluskid/fitting-random-labels
from torchvision import datasets, transforms
import torch
import numpy as np
import time

class MNISTRandomLabels(datasets.MNIST):
    """MNIST dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(MNISTRandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets if self.train else self.test_labels)
        np.random.seed(int(time.time()))
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]

        if self.train:
            self.targets = labels
        else:
            self.targets = labels
class CIFAR100RandomLabels(datasets.CIFAR100):
    """cifar-100 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 100. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=100, **kwargs):
        super(CIFAR100RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets if self.train else self.test_labels)
        np.random.seed(int(time.time()))
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]

        if self.train:
            self.targets = labels
        else:
            self.targets = labels

class CIFAR10RandomLabels(datasets.CIFAR10):
    """cifar10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets if self.train else self.test_labels)
        np.random.seed(int(time.time()))
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]

        if self.train:
            self.targets = labels
        else:
            self.targets = labels
            
def get_data_loader(name, batch_size, num_samples=None, corrupt_prob = 0):
    """ get test and train dataloaders
    Params
    -----
    name: the name of the dataset. Choices are: 'cifar10', 'mnist', and 'cifar100'
    batch_size: int
    The size of the batch.
    num_samples: int
    Default None. The number of training samples to use.
    corrupt_prob: float between 0 and 1
    Default 0. The probability of a label being random. 
    """
    if name == 'cifar10':
        train_dataset = CIFAR10RandomLabels(root='./data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),
                                                   download=True, corrupt_prob = corrupt_prob)
        test_dataset = datasets.CIFAR10(root='./data', 
                                              train=False, download=True,
                                              transform=transforms.ToTensor())
    elif name == 'mnist':
        train_dataset = MNISTRandomLabels(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True, corrupt_prob = corrupt_prob)
        test_dataset = datasets.MNIST(root='./data', 
                                          train=False, download=True,
                                          transform=transforms.ToTensor())
    elif name == 'cifar100':
        
        train_dataset = CIFAR100RandomLabels(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True, corrupt_prob = corrupt_prob)
        test_dataset = datasets.CIFAR100(root='./data', 
                                          train=False, download=True,
                                          transform=transforms.ToTensor())

    if num_samples == None:
        num_samples = len(train_dataset)
     
    # in case we want to train on part of the training set instead of all
    my_train_dataset, rest_train_dataset = torch.utils.data.random_split(dataset=train_dataset, lengths=[num_samples,len(train_dataset)-num_samples])

    
    # Data loader
    my_train_loader = torch.utils.data.DataLoader(dataset=my_train_dataset, num_workers = 11,
                                               batch_size=batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers = 11,
                                              batch_size=batch_size, 
                                              shuffle=True)
    return my_train_loader, test_loader, my_train_dataset
