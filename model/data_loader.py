from matplotlib.pylab import cond
import pytorch_lightning as pl
import numpy as np
import torch
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import random

class MaizeDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        img_size: int,
        split_dir: str,
        debug,
    ):
        super(MaizeDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4
        
        self.train_data = occupancy_field_Dataset(self.data_dir, [os.path.join(split_dir, 'train_files.txt')],img_size)
        self.test_data = occupancy_field_Dataset(self.data_dir, [os.path.join(split_dir, 'test_files.txt')], img_size)

    def dataloader(self, train=True): 
        if train:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
            )
        
class UnconditionalMaizeDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        img_size: int,
        split_dir: str,
        debug,
    ):
        super(UnconditionalMaizeDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4
        
        self.data = occupancy_field_Dataset(self.data_dir, [os.path.join(split_dir, 'train_files.txt'), os.path.join(split_dir, 'test_files.txt')],
                                            img_size, condition=False)

    def dataloader(self, train=True): 
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

class occupancy_field_Dataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 file_list: list,
                 img_size: int = 448,
                 condition: bool = True
    ):
        super().__init__()
        self.condition = condition
        self.img_size = img_size
        self.file_lists = []
        for fl in file_list:
            with open(fl, 'r', encoding='utf-8') as f:
                self.file_lists.extend(f.read().splitlines())
        self.img_folder = os.path.join(data_folder, "img")
        self.vxl_folder = os.path.join(data_folder, "vxl")
        assert os.path.exists(self.img_folder), f"path '{self.img_folder}' does not exists."
        assert os.path.exists(self.vxl_folder), f"path '{self.vxl_folder}' does not exists."
        if condition:
            self.img_list = {i: [os.path.join(self.img_folder, k) for k in os.listdir(self.img_folder) if k.startswith(i)]
                                            for i in self.file_lists}
    
        self.vxl_list = {i: self.load_voxel(os.path.join(self.vxl_folder, i + '.npy')) for i in self.file_lists}

        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_voxel(self, path):
        voxel = np.load(path)
        leaf_index = np.where(voxel > 0.5)
        voxel[leaf_index] = 1
        return voxel
    
    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        data = {}
        id = self.file_lists[index]
        if self.condition:
            img = random.choice(self.img_list[id])
            img = Image.open(img).convert("RGB")
            data["img"] = self.transform(img)
        else:
            data["img"] = torch.zeros(3, self.img_size, self.img_size)
        voxel = self.vxl_list[id]
        data["voxel"] = voxel
        # print(data["img"].min(), data["img"].max())
        # transforms.ToPILImage()(data["img"]).show()
        # print(self.img_list[index], self.vxl_list[index])
        return data

# loader = occupancy_field_Dataset("/datasets/joe/dataset/maize", ['/home/yuezhuo/skeleton/DiffTreeVxl/dataset/maize/train_files.txt'])
# data = loader.__getitem__(0)
# print(data["voxel"].max(), data["voxel"].min())
# print(data["voxel"].shape)