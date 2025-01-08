# MediaPipe Tutorial for Beginners

Welcome to this tutorial on **MediaPipe**! MediaPipe is an amazing library created by Google that helps us do cool things with computer vision, like detecting hands, faces, or even the whole body using a webcam or images.

Let’s get started step-by-step. By the end, you'll learn how to detect hands with MediaPipe.

---

## Prerequisites

Before starting, make sure you have Python installed on your computer. You can check by running this command in your terminal or command prompt:

```bash
python --version
```

If you don’t have Python, [download and install it here](https://www.python.org/downloads/).

---

## Step 1: Install MediaPipe and OpenCV

We need two libraries: **MediaPipe** and **OpenCV**. MediaPipe will handle hand detection, and OpenCV will help us use the webcam and display images.

Run this command to install both:

```bash
pip install mediapipe opencv-python
```

---

## Step 2: Create a Python File

1. Open your code editor (e.g., VS Code or PyCharm).
2. Create a new file and name it `hand_detection.py`.

---

## Step 3: Basic Setup for MediaPipe

Here’s the code to get started with hand detection:

```python
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize Hands module
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally for a selfie view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe works with RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow("Hand Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
```

---

## Step 4: Run the Code

1. Save the file.
2. Open your terminal or command prompt in the folder where the file is saved.
3. Run the script:

```bash
python hand_detection.py
```

---

## What Happens Next?

1. A window will pop up showing the video feed from your webcam.
2. Move your hand in front of the camera—you should see lines and points drawn on your hand!

---

## Step 5: Customize the Code

### Change Detection Confidence
You can adjust the detection confidence by changing this line:

```python
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
```

Increase `0.5` to a higher number (e.g., `0.8`) for stricter detection or lower it for more flexibility.

### Add FPS Counter
Want to see how fast your program is running? Add this code inside the `while` loop to calculate frames per second:

```python
import time
prev_time = 0

while cap.isOpened():
    # Your existing code here...

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
```

---

## What's Next?

Here are some fun things to try:
1. **Face Detection**: MediaPipe can detect faces too! Replace `mp.solutions.hands` with `mp.solutions.face_detection`.
2. **Pose Estimation**: Detect body movements using `mp.solutions.pose`.

You can explore the [MediaPipe Documentation](https://google.github.io/mediapipe/) to learn about more features.

---

## Troubleshooting

- **No webcam detected**: Check if your webcam is properly connected.
- **Error with `mediapipe`**: Make sure you installed the library using `pip install mediapipe`.
- **Slow performance**: Try running the code on a computer with a better GPU or processor.

---

That’s it! You’ve just learned how to use MediaPipe for hand detection. Keep experimenting and having fun!

