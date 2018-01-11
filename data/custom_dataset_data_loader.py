import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os

def CreateDataset(dataroot,dataset_mode='2afc',load_size=64):
    dataset = None
    if dataset_mode=='2afc': # human judgements
        from dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized."%self.dataset_mode)

    dataset.initialize(dataroot,load_size=load_size)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, datafolder, dataroot='./bapd',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        BaseDataLoader.initialize(self)
        self.dataset = CreateDataset(os.path.join(dataroot,datafolder),dataset_mode=dataset_mode,load_size=load_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
