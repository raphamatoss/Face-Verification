# Face-Verification
I'm currently undertaking a scientific initiation project in computer vision at my university. My goal is to develop a facial verification model capable of achieving strong performance metrics on the Labeled Faces in the Wild (LFW) Dataset.

### Pipelines.py
Contains the main classes and methods responsible for the processing of the images and for inference. There is the PreprocessPipeline which preprocess the image, detecting the face in it and applying the requeired normalization before sending it to the embedder. 
There's also the ExctractionPipeline, which consists of the inference step on the embedder, it requires a preprocessed face and returns a 521-D vector. 
Finally there are functions especialized in performing the cosine similarity calculation.

### Dataset_reader.py
It only process the dataset folder, turning it into a dict of key: label and value: list of embeddings, the list of embeddings is the list of the embeddings of each image of the person in the dataset.

### Evaluation.py
It performs the evaluation of the model in the LFW_Dataset based on the mismatch_pairs.csv and match_pairs.csv.

# Credits 
This work utilizes a base pipeline adapted from @alaasweed's Kaggle repository, with notable extensions including native PyTorch Tensor support and compatibility across both TensorFlow and PyTorch MTCNN frameworks.
