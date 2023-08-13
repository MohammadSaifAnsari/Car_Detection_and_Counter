import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# for Webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # frameWidth
# cap.set(4, 480)  # frameHeight

cap = cv2.VideoCapture("Video/cars.mp4")

model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# For using mask Image
maskImg = cv2.imread("mask1.png")

# For Tracking
track = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Dimension of line x1,y1,x2,y2
limits = [225, 200, 433, 200]

# For Calculating no of cars cross the line
totalCount = []
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, maskImg)

    results = model(imgRegion, stream=True)

    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # for opencv bounding box
            # cv2.rectangle(img, (x1, y1),(x2, y2), (255,0,255),3)

            # for cvzone bounding box
            # w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=5)

            # Confidence Level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "motorbike" or currentClass == "bus" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                # scale=0.5, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentArray))

    resultTrack = track.update(detection)
    # For Displaying Line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultTrack:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # For putting circle at the center of bounding box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # For setiing range to detect car
        if limits[0] < cx < limits[2] and limits[1] - 25 < cy < limits[3] + 25:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    # For showing counter
    cvzone.putTextRect(img, f'count :{len(totalCount)}', (50, 50))

    cv2.imshow("Car Counter", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
