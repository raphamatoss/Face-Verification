import facenet_pytorch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import numpy as np
import mtcnn
import torch
import cv2

class preprocessPipeline:
    def __init__(self, detector, cmap="rgb", device="cpu"):
        self.detector = detector
        self.target_size = (224, 224)
        self.cmap = cmap
        self.device = device

    def load_image(self, image_input):
        if isinstance(image_input, str) or isinstance(image_input, Path):
            image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        elif isinstance(image_input, torch.Tensor):
            image = image_input.permute(1, 2, 0).cpu().numpy()
        else:
            raise Exception(f'Invalid image type: {type(image_input)}')

        if self.cmap == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.merge([image, image, image])

        return image

    def resize_and_normalize(self, image):
        if image is None or image.size == 0:
            raise ValueError("Empty image caught by resize_and_normalize")

        resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        normalized_image = resized_image.astype(np.float32) / 255.0
        return normalized_image

    def detect_face(self, image):
        if isinstance(self.detector, facenet_pytorch.MTCNN): #pytorch based mtcnn
            image = Image.fromarray(image)  # transforms numpy array into a PIL image
            self.detector.to(self.device)

            faces = self.detector(image)
            if faces is not None: #largest is selected
                face_image = faces
                return face_image;
            else:
                raise Exception('No face detected')
        elif isinstance(self.detector, mtcnn.MTCNN): #tensorflow based mtcnn
            faces = self.detector.detect_faces(image)
            if faces:
                face = faces[0]
                x, y, w, h = face['box']
                x, y = abs(x), abs(y)
                face_image  = image[y:y+h, x:x+w]
                return face_image
            else:
                raise Exception('No face detected')
        else:
            raise Exception(f'Invalid detector type: {type(self.detector)}')

    def preprocess(self, image):
        image = self.load_image(image)
        face = self.detect_face(image)
        if isinstance(face, torch.Tensor):
            face = face.permute(1, 2, 0).cpu().numpy()
        return self.resize_and_normalize(face)

class extractionPipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = v2.Compose([
            v2.Lambda(lambda x: Image.fromarray((x * 255).astype(np.uint8)) if isinstance(x,
                                                                                          np.ndarray) and x.dtype == np.float32 else Image.fromarray(
                x)),
            v2.Resize((160, 160), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
        ])
    def extract(self, image_input):
        image_tensor = self.transform(image_input).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model(image_tensor).squeeze().cpu().numpy()
        return features


class fullPipeline:
    def __init__(self, detector, device="cpu"):
        self.device = device
        self.preprocess = preprocessPipeline(detector, device)
        self.extraction = extractionPipeline(device)

    def process(self, input):
        preprocessed = self.preprocess.preprocess(input)
        embedding = self.extraction.extract(preprocessed)
        return embedding

# COSINE SIMILARITY FUNCTIONS --------------------------------------

def l2_normalization(input):
    if isinstance(input, torch.Tensor):
        norm = torch.linalg.norm(input)
        return input if norm == 0 else input / norm
    if isinstance(input, np.ndarray):
        norm = np.linalg.norm(input)
        return input if norm == 0 else input / norm
    raise Exception(f'Invalid input type: {type(input)}')


def cos_similarity(input1, input2):
    if isinstance(input1, torch.Tensor):
        input1 = l2_normalization(input1)
        input2 = l2_normalization(input2)
        cos_sim = torch.dot(input1, input2).item()
    else:
        input1 = l2_normalization(input1)
        input2 = l2_normalization(input2)
        cos_sim = np.dot(input1, input2)
    return cos_sim

def cos_sim_to_percentage(cosine_similarity):
    cosine_similarity = np.clip(cosine_similarity, 0, 1)
    return 100 * cosine_similarity