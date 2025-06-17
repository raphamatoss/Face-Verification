from tqdm import tqdm
from pathlib import Path
import pipelines as p
import csv
import glob
import os

def parse_people(csv_path):
    people = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[1].strip().isdigit():
                continue
            name = row[0].strip()
            images_number = int(row[1].strip())
            people[name] = images_number
    return people

def build_dataset(dataset_path):
    people_path = Path(dataset_path).joinpath('people.csv')
    people = parse_people(people_path)
    funneled_path = Path(dataset_path).joinpath('lfw-deepfunneled').joinpath('lfw-deepfunneled')
    map = {}
    for person in people.keys():
        person_dir = os.path.join(funneled_path, person)
        if os.path.isdir(person_dir):
            images_path = glob.glob(os.path.join(person_dir, '*.jpg'))
            images_path.sort()
            map[person] = images_path
        else:
            print(f'{person}''s directory not found.')
    return map

def transform_dataset(dataset, detector, cmap, device):
    preprocess_pipeline = p.preprocessPipeline(detector, cmap, device)
    extraction_pipeline = p.extractionPipeline(device)

    map = {}
    for key in tqdm(dataset.keys()):
        faces = dataset[key]
        if faces is not None:
            face_features = []
            for face in faces:
                try:
                    processed_image = preprocess_pipeline.preprocess(face)
                    features = extraction_pipeline.extract(processed_image)
                    face_features.append(features)
                except Exception as e:
                    print(f"{key}: {e}")
            map[key] = face_features
    return map