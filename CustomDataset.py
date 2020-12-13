from torch.utils.data import Dataset
import numpy as np
from eda import prep_dataset

class CustomDataset(Dataset):
    def __init__(self, device):
        self.dataset = np.load('data_pub.zip')
        self.device = device


    def __len__(self):
        return len(self.dataset)-2

    def __getitem__(self, item):
        try:
            sample = self.dataset[item]
        except:
            sample = self.dataset[item+1].shape
        return prep_dataset(sample, self.device)
