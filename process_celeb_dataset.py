import os
from sklearn.model_selection import train_test_split
import shutil


def split_data_set(source_path="./data/CelebAMask-HQ/CelebA-HQ-img/"):
    files = os.listdir(source_path)
    files = [fil.split('.')[0] for fil in files]
    xtrain, xtest = train_test_split(files, test_size=0.3, random_state=42)
    if not os.path.exists("./data/CelebAMask-HQ/training"):
        os.mkdir("./data/CelebAMask-HQ/training")
    if not os.path.exists("./data/CelebAMask-HQ/test"):
        os.mkdir("./data/CelebAMask-HQ/test")

    for image in xtrain:
        shutil.copy(source_path + image + ".jpg", "./data/CelebAMask-HQ/training/")
    for image in xtest:
        shutil.copy(source_path + image + ".jpg", "./data/CelebAMask-HQ/test/")


if __name__ == "__main__":
    split_data_set()
