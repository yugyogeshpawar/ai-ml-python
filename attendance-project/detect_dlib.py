import cv2
import dlib
import numpy as np
import pickle
import os

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load trained encodings
encodings_dir = 'trained_encodings_dlib'
encodings = {}
for file in os.listdir(encodings_dir):
    if file.endswith('.pkl'):
        name = os.path.splitext(file)[0]
        with open(f"{encodings_dir}/{file}", "rb") as f:
            encodings[name] = pickle.load(f)

# Function to recognize faces
def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = shape_predictor(gray, face)
        encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))

        # Compare with known encodings
        name = "Unknown"
        distances = []
        for known_name, known_encoding in encodings.items():
            distance = np.linalg.norm(known_encoding - encoding)
            distances.append((distance, known_name))
        
        # Identify the person with the minimum distance
        if distances:
            distances.sort()
            best_match_distance, best_match_name = distances[0]
            if best_match_distance < 0.6:  # Set threshold
                name = best_match_name

        # Draw a box around the face
        left, top, right, bottom = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Start video capture and face recognition
video_capture = cv2.VideoCapture(1)
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    recognize_faces(frame)
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
