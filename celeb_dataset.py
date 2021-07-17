import torch
import os
import numpy as np
import time
from PIL import Image
import torchvision.transforms as tforms
from torch.utils.data import Dataset

class CelebAHQ(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None, ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_files_names = os.listdir(root)
        self.root_dir = root
        self.transform = transform
        

    def __len__(self):
        return len(self.training_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.training_files_names[idx])
        

        with open(img_name, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        return image, np.zeros_like(image)
    
if __name__=="__main__":
    train_set = CelebAHQ("/HPS/CNF/work/ffjord-rnode_first_atempt/data/CelebAMask-HQ/training/",
                         transform=tforms.Compose([tforms.Resize(32), tforms.RandomHorizontalFlip()]))
    test_set = CelebAHQ("/HPS/CNF/work/ffjord-rnode_first_atempt/data/CelebAMask-HQ/test/",
                         transform=tforms.Compose([tforms.Resize(32)]))
    im_dim = 3
    im_size = 32
    def fast_collate(batch):

        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        
        w = imgs[0].shape[1]
        h = imgs[0].shape[2]
        
        tensor = torch.zeros( (len(imgs), im_dim, im_size, im_size), dtype=torch.uint8 )
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            tens = torch.from_numpy(nump_array)
            if(nump_array.ndim < 3):
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)
            tensor[i] += torch.from_numpy(nump_array)

        targets = torch.zeros( (len(imgs), im_dim, im_size, im_size), dtype=torch.uint8)
        return tensor, targets
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=8, #shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=fast_collate
    )
    x, label = next(iter(train_loader))
    print(x.shape, label.shape)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_set,  batch_size=8, #shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=fast_collate)
    