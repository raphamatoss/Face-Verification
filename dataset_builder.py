import torch
import glob
import os
from torch.utils.data import DataLoader, Dataset
import pipelines as p
from tqdm.auto import tqdm

class LFWDataset(Dataset):
    def __init__(self, dataset, detector, device=torch.device('cpu')):
        self.dataset_path = dataset
        self.data = []
        self.device = device
        self.pipeline = p.fullPipeline(detector, device)

        people = glob.glob(os.path.join(self.dataset_path, '*'))
        for person in tqdm(people):
            label = os.path.basename(person)
            print(label)
            for img in glob.glob(os.path.join(person, '*.jpg')):
                embedding = self.pipeline.process(img)
                self.data.append((embedding, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label = self.data[idx]
        return embedding, label