import cv2
import dlib
import numpy as np
import os
import pickle

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory containing the dataset
dataset_dir = 'dataset'

# Directory to save trained encodings
encodings_dir = 'trained_encodings_dlib'
os.makedirs(encodings_dir, exist_ok=True)

def process_image(image_path):
    """Processes an image to extract face encodings."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print(f"No face detected in image: {image_path}")
        return None

    # Use the first face detected
    face = faces[0]
    shape = shape_predictor(gray, face)
    encoding = np.array(face_rec_model.compute_face_descriptor(image, shape, 1))
    return encoding

def train_on_dataset():
    """Trains on the dataset and saves average encodings for each person."""
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"Processing images for {person_name}...")
        encodings = []

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            encoding = process_image(image_path)
            if encoding is not None:
                encodings.append(encoding)

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            with open(f"{encodings_dir}/{person_name}.pkl", "wb") as f:
                pickle.dump(avg_encoding, f)
            print(f"Saved encoding for {person_name}.")
        else:
            print(f"No valid encodings found for {person_name}.")

if __name__ == "__main__":
    train_on_dataset()
    print("Training on dataset completed.")
