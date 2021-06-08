import torch
import os
import numpy as np
import torchvision.transforms as tforms
from torch.utils.data import Dataset


class CelebDataset(Dataset):

    def __init__(self, root, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_files_names = np.unique([fil.split("_")[0] for fil in os.listdir(root)])
        self.root_dir = root
        self.transform = transform

    def __len__(self):
        return len(self.training_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.training_files_names[idx])
        loaded_ = np.load(img_name+"_train.npz")
        train_image = loaded_["a"]

        loaded = np.load(img_name + "_label.npz")
        label = loaded["a"]
        return train_image, label


if __name__ == "__main__":
    train_set = CelebDataset("./data/CelebAMask-HQ/training_sets/2/", )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=8,  # shuffle=True,
        num_workers=8, pin_memory=True)

    x,y = next(iter(train_loader))
    print(x.shape, y.shape)
    test_set = CelebDataset("./data/CelebAMask-HQ/test_sets/3/",  transform=tforms.Compose([tforms.ToTensor()]))
