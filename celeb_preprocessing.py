import torch
import os
import numpy as np
from skimage import io
import torchvision.transforms as tforms


class CelebPreprocessing():
    """Face Landmarks dataset."""

    def __init__(self, root, save_files_path, transform=None, number_of_downscales=4, downscale_factor=2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_files_names = os.listdir(root)
        self.root_dir = root
        self.save_files_path = save_files_path
        self.transform = transform
        self.downscale_factor = downscale_factor
        self.number_of_downscales = number_of_downscales

    def len(self):
        return len(self.training_files_names)

    def getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.training_files_names[idx])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
            image = torch.unsqueeze(image, 0)
        # print(image.size())
        for i in range(self.number_of_downscales + 1):
            if not os.path.exists(self.root_dir + self.save_files_path + str(i)):
                os.mkdir(self.root_dir + self.save_files_path + str(i))
        for i in range(self.number_of_downscales):

            batch_size, in_channels, in_height, in_width = image.size()
            out_channels = in_channels * (self.downscale_factor ** 2)

            out_height = in_height // self.downscale_factor
            out_width = in_width // self.downscale_factor
            input_view = image.contiguous().view(
                batch_size, in_channels, out_height, self.downscale_factor, out_width, self.downscale_factor
            )

            output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
            output = output.view(batch_size, out_channels, out_height, out_width)
            d = output.size(1) // 2

            if out_height >= 4:
                image = output[:, :d]
                np.savez_compressed(
                    self.root_dir + self.save_files_path + str(i) + "/" + self.training_files_names[idx].split('.')[0]
                    + "_label",
                    a=torch.squeeze(output[:, :d]).numpy())
                np.savez_compressed(
                    self.root_dir + self.save_files_path + str(i + 1) + "/" + self.training_files_names[idx].split('.')[
                        0]
                    + "_train",
                    a=torch.squeeze(output[:, :d]).numpy())
            else:
                image = output
                np.savez_compressed(
                    self.root_dir + self.save_files_path + str(i) + "/" + self.training_files_names[idx].split('.')[0]
                    + "_label",
                    a=torch.squeeze(output).numpy())
            # print("image : ", image.size())
        # return torch.squeeze(image)


if __name__ == "__main__":
    if not os.path.exists("./data/CelebAMask-HQ/training_sets/"):
        os.mkdir("./data/CelebAMask-HQ/training_sets/")
    if not os.path.exists("./data/CelebAMask-HQ/test_sets/"):
        os.mkdir("./data/CelebAMask-HQ/test_sets/")
    train_set = CelebDataset("./data/CelebAMask-HQ/training/", "../training_sets/",
                             transform=tforms.Compose([
                                 tforms.ToPILImage(),
                                 tforms.Resize(32),
                                 tforms.RandomHorizontalFlip(),
                                 tforms.ToTensor()
                             ]))
    for i in range(train_set.len()):
        if i % 1000 == 0:
            print(i)
        x = train_set.getitem(i)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_set, batch_size=1,  # shuffle=True,
    #     num_workers=8, pin_memory=True)

    # x = next(iter(train_loader))
    test_set = CelebDataset("./data/CelebAMask-HQ/test/", "../test_sets/", transform=tforms.Compose([
        tforms.ToPILImage(),
        tforms.Resize(32),
        tforms.ToTensor()
    ]))

    for i in range(test_set.len()):
        if i % 1000 == 0:
            print(i)
        x = test_set.getitem(i)
