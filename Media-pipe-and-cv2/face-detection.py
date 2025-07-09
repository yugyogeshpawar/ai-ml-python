import cv2
import mediapipe as mp

# Initialize the face detection module from MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open a connection to the webcam
cap = cv2.VideoCapture(1)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5) 
# Set up the face detection with a detection confidence level
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame from the camera.")
            continue
        
        # Convert the image color format from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        results = face_detection.process(image)
        
        # Convert the image color format back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face detection annotations if faces are detected
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        
        # Display the frame with detections
        cv2.imshow('Face Detection', image)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# # Release the resources
cap.release()
cv2.destroyAllWindows()
