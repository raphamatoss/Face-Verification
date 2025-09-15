import dataset_builder as db
import torch
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, random_split
import os

import evaluation as ev

f = open('dataset_path.txt')
dataset_path = f.read()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(image_size=160,
                 margin=14,
                 device=device,
                 post_process=False,
                 selection_method='center_weighted_size')

if not os.path.exists('LFWDataset.pt'):
    LFWDataset = db.LFWDataset(dataset_path, detector, device, 2)
    torch.save(LFWDataset, 'LFWDataset.pt')
    print('LFWDataset.pt saved')
else:
    LFWDataset = torch.load('LFWDataset.pt', weights_only=False)
    print('LFWDataset.pt loaded')

print(LFWDataset.usage_percentage, LFWDataset.num_classes)

LFW_loader = torch.utils.data.DataLoader(LFWDataset, batch_size=1, shuffle=True)
ev.identification_evaluation(LFW_loader)


# LFWImageFolder = db.get_imageFolder(dataset_path, detector, device)
# print("ImageFolder loaded -----------")
# print(LFWImageFolder + "\n")
# class_names, class_dict = LFWImageFolder.classes, LFWImageFolder.class_to_idx
# print(len(class_names))
#
# LFW_loader = DataLoader(LFWImageFolder, batch_size=1, shuffle=True, num_workers=os.cpu_count())
# print("DataLoader loaded -----------")
