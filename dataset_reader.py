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

def build_dataset(dataset_path, people_file_path):
    people = parse_people(people_file_path)
    map = {}
    for person in people.keys():
        person_dir = os.path.join(dataset_path, person)
        if os.path.isdir(person_dir):
            images_path = glob.glob(os.path.join(person_dir, '*.jpg'))
            images_path.sort()
            map[person] = images_path
        else:
            print(f'{person}''s directory not found.')
    return map
