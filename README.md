# Face Recognition Using Deep Learning

This project implements a deep learning-based face recognition system using OpenCV, dlib, and scikit-learn. It provides scripts for extracting facial embeddings from images, training a face recognition model, and recognizing faces in both images and real-time video streams.

## Features

- Face detection using OpenCV's deep learning face detector.
- Face embedding extraction with OpenFace.
- Face recognition using an SVM classifier.
- Recognize faces in static images or live video from a webcam.

## Project Structure

```
face-recognition-using-deep-learning-master/
│
├── dataset/                       # Folder containing subfolders of images for each person
├── face_detection_model/          # Contains deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel
├── output/                        # Stores embeddings.pickle, recognizer.pickle, le.pickle
├── extract_embeddings.py          # Extracts face embeddings from dataset images
├── train_model.py                 # Trains the SVM face recognizer
├── recognize_image.py             # Recognizes faces in a single image
├── recognize_video.py             # Recognizes faces in real-time video
├── openface_nn4.small2.v1.t7      # Pre-trained OpenFace embedding model
├── README.md                      # Project documentation
└── LICENSE                        # License information  
```

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- imutils
- numpy
- scikit-learn
- dlib (optional, for dataset preparation)

Install dependencies with:
```bash
pip install opencv-python imutils numpy scikit-learn
```

## Usage

### 1. Prepare Dataset

Organize your dataset as:
```
dataset/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        ...
```

### 2. Extract Embeddings

Extract 128-d face embeddings from your dataset:
```bash
python extract_embeddings.py
```

### 3. Train the Recognizer

Train the SVM classifier on the extracted embeddings:
```bash
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
```

### 4. Recognize Faces in an Image

Recognize faces in a single image:
```bash
python recognize_image.py --image path/to/image.jpg
```

### 5. Recognize Faces in Real-Time Video

Recognize faces from your webcam:
```bash
python recognize_video.py
```

## Pre-trained Models

- Download the [OpenFace model](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7) and place it in the project root.
- Download the [Caffe face detector model](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in `face_detection_model/`.

## License

This project is for educational purposes.

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [OpenFace](https://cmusatyalab.github.io/openface/)
- [imutils](https://github.com/jrosebr1/imutils)