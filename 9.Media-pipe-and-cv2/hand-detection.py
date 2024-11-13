import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open a connection to the webcam
cap = cv2.VideoCapture(1)

# Set up the hand detection with a detection confidence level
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame from the camera.")
            continue
        
        # Convert the image color format from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Process the frame to detect hands
        results = hands.process(image)
        
        # Convert the image color format back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw hand landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the frame with hand landmarks
        cv2.imshow('Hand Detection', image)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the resources
cap.release()
cv2.destroyAllWindows()
