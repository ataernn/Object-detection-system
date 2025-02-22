import cv2
import tensorflow as tf
import numpy as np

model_path = ".\\ssd_mobilenet_v2_coco_2018_03_29\\saved_model" 
model = tf.saved_model.load(model_path)

infer = model.signatures['serving_default']

class_map = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorbike", 5: "aeroplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow",
    21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack",
    26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee",
    31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat",
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 
    40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 
    45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 
    51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 
    56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 
    61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 
    66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 
    71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 
    76: "vase", 77: "cellphone", 78: "teddy bear", 79: "hair drier", 80: "toothbrush"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = infer(input_tensor)

    num_detections = int(detections['num_detections'][0].numpy())

    for i in range(num_detections):
        bbox = detections['detection_boxes'][0][i].numpy()
        class_id = int(detections['detection_classes'][0][i].numpy())
        score = detections['detection_scores'][0][i].numpy()

        if score > 0.5:
            (h, w, _) = frame.shape
            ymin, xmin, ymax, xmax = bbox
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)

            class_name = class_map.get(class_id, "Unknown")

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} - {score:.2f}', (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
