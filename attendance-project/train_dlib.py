import cv2
import dlib
import numpy as np
import os
import pickle

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory to save trained encodings
encodings_dir = 'trained_encodings_dlib'
os.makedirs(encodings_dir, exist_ok=True)

def capture_and_train(name):
    video_capture = cv2.VideoCapture(1)
    encodings = []

    print(f"Starting training for {name}. Please look at the camera.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            shape = shape_predictor(gray, face)
            # Compute the face encoding
            encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))
            encodings.append(encoding)
            print(f"Captured encoding {len(encodings)} for {name}.")

        # Display the frame
        cv2.imshow("Training - Press 'q' when done", frame)

        # Press 'q' to stop capturing images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Limit to 10 encodings for each person
        if len(encodings) >= 10:
            print(f"Captured enough encodings for {name}.")
            break

    # Save the average encoding for the person
    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        with open(f"{encodings_dir}/{name}.pkl", "wb") as f:
            pickle.dump(avg_encoding, f)
        print(f"Training for {name} completed and saved.")

    # Release the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Main loop to train multiple people
while True:
    name = input("Enter the name of the person to train (or type 'exit' to quit): ")
    if name.lower() == 'exit':
        print("Exiting training.")
        break

    capture_and_train(name)
    print("Training is complete. Type 'next' to train another person or 'exit' to stop.")
