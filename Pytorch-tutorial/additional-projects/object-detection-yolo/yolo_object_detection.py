# yolo_object_detection.py

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Loads a pre-trained YOLOv5 model from PyTorch Hub and uses it to
    detect objects in a sample image.
    """
    # --- 1. Load Pre-trained YOLOv5 Model ---
    print("--- Loading YOLOv5 model from PyTorch Hub ---")
    # This will download the model and its dependencies on the first run.
    # 'yolov5s' is the small version of the model.
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and git installed.")
        return

    # You can set model parameters like confidence threshold and IoU threshold
    model.conf = 0.4  # Confidence threshold
    model.iou = 0.5   # Intersection over Union (IoU) threshold for NMS

    print("--- Model Loaded Successfully ---")

    # --- 2. Prepare an Image ---
    # You can use a local file path or a URL.
    # Let's use a well-known image from the COCO dataset for this example.
    img_path = 'https://ultralytics.com/images/zidane.jpg'
    # Or use a local file:
    # img_path = 'path/to/your/image.jpg'

    # --- 3. Perform Inference ---
    print(f"\n--- Performing inference on {img_path} ---")
    results = model(img_path)

    # --- 4. Process and Visualize Results ---
    # The `results` object contains a lot of information.
    # We can easily convert it to a pandas DataFrame for inspection.
    print("\n--- Detection Results (Pandas DataFrame) ---")
    results_df = results.pandas().xyxy[0]  # xyxy format gives (xmin, ymin, xmax, ymax)
    print(results_df)

    # The `results` object also has a `render()` method that returns the
    # image with bounding boxes and labels drawn on it.
    rendered_image = results.render()[0] # render() returns a list of images

    # Convert the rendered image from BGR (OpenCV default) to RGB (matplotlib default)
    rendered_image_rgb = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

    # Display the image
    print("\n--- Displaying Image with Detections ---")
    plt.figure(figsize=(12, 12))
    plt.imshow(rendered_image_rgb)
    plt.title("YOLOv5 Object Detection")
    plt.axis('off')
    plt.show()

    # --- Alternative: Manual Drawing (for more control) ---
    # If you want more control over how the boxes are drawn.
    print("\n--- Manually Drawing Bounding Boxes ---")
    
    # Load the original image again using OpenCV
    # We need to download it first if it's a URL
    try:
        import requests
        from PIL import Image
        import io
        response = requests.get(img_path)
        img = Image.open(io.BytesIO(response.content))
        original_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except:
        print("Could not download image, skipping manual drawing.")
        return

    # Loop through the detected objects in the DataFrame
    for index, row in results_df.iterrows():
        # Get coordinates, class name, and confidence
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        
        # Draw rectangle
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert for display
    manual_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(manual_image_rgb)
    plt.title("YOLOv5 Object Detection (Manual Drawing)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # This script requires:
    # pip install torch torchvision opencv-python matplotlib pandas requests
    main()
