import cv2
import dlib
import numpy as np
import os
import pickle

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory containing trained encodings
encodings_dir = 'trained_encodings_dlib'

def load_encodings():
    """Loads the saved encodings from the directory."""
    encodings = {}
    for file_name in os.listdir(encodings_dir):
        if file_name.endswith(".pkl"):
            person_name = file_name[:-4]
            with open(os.path.join(encodings_dir, file_name), "rb") as f:
                encodings[person_name] = pickle.load(f)
    return encodings

def recognize_face(frame, known_encodings):
    """Recognizes a face in the given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        shape = shape_predictor(gray, face)
        # Compute the face encoding
        encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape, 1))

        # Compare with known encodings
        min_distance = float("inf")
        name = "Unknown"

        for person_name, person_encoding in known_encodings.items():
            distance = np.linalg.norm(encoding - person_encoding)
            if distance < min_distance and distance < 0.6:  # Threshold for recognition
                min_distance = distance
                name = person_name

        # Draw a rectangle around the face and label it
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def main():
    """Main function for real-time face detection and recognition."""
    print("Loading encodings...")
    known_encodings = load_encodings()
    print("Encodings loaded.")

    video_capture = cv2.VideoCapture(1)
    print("Starting video stream. Press 'q' to exit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = recognize_face(frame, known_encodings)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
