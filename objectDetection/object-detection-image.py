import cv2

from ultralytics import YOLO

yolo = YOLO('yolov8s.pt')

videoCap = cv2.VideoCapture(0)

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


# img = cv2.imread("image.jpg")

results = yolo.track("istockphoto-1397843652-612x612.jpg", stream=False)

print(results)

# Load the image for drawing
image = cv2.imread("istockphoto-1397843652-612x612.jpg")

for result in results:
    classes_names = result.names
    for box in result.boxes:
        if box.conf[0] > 0.4:
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            class_name = classes_names[cls]
            colour = getColours(cls)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

# Save the annotated image
cv2.imwrite("output_image.jpg", image)
