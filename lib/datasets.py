import os
import torch
import torchvision

class Dataset(object):

    def __init__(self, loc, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(loc)
        if in_mem: self.dataset = self.dataset.float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        if not self.in_mem: x = x.float().div(255)
        x = self.transform(x) if self.transform is not None else x
        return x, 0

# To acquire these datasets, follow instructions in ../preprocessing/

class Imagenet64(torchvision.datasets.ImageFolder):

    def __init__(self, train=True, transform=None, root='./data/'):
        self.train_loc = os.path.join(root, 'imagenet64/train/')
        self.test_loc = os.path.join(root, 'imagenet64/val/')
        return super().__init__(self.train_loc if train else self.test_loc, transform=transform)


class CelebAHQ(Dataset):

    def __init__(self, train=True, transform=None, root='./data/'):
        self.train_loc = os.path.join(root,'celebahq/celeba256_train.pth')
        self.test_loc = os.path.join(root,'celebahq/celeba256_validation.pth')
        return super(CelebAHQ, self).__init__(self.train_loc if train else self.test_loc, transform=transform, in_mem=False)
