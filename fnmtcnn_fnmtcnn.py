import dataset_reader as dr
from pathlib import Path
from facenet_pytorch import MTCNN
import evaluation as ev
import pickle
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(image_size=160,
                 margin=14,
                 device=device,
                 post_process=False,
                 selection_method='center_weighted_size')

f = open("dataset_path.txt")
path = f.read()
dataset = dr.build_dataset(path)

transformed_dataset_path = Path("fnmtcnn_fnmtcnn_rgb.pkl")
if transformed_dataset_path.exists():
    print("Dataset already exists. Loading...")
    with open(transformed_dataset_path, "rb") as f:
        transformed_dataset = pickle.load(f)
else:
    print("Transforming dataset with MTCNN from facenet and RGB color map...")
    transformed_dataset = dr.transform_dataset(dataset, detector, 'rgb', device)
    print("Saving transformed dataset...")
    with open(transformed_dataset_path, "wb") as f:
        pickle.dump(transformed_dataset, f)

ev.multi_eval(transformed_dataset, path, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])