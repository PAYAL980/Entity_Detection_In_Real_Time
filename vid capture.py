# import necessary packages
import cv2
import numpy as np

# load the model trained using YOLO
network = cv2.dnn.readNet('yolo.weights', 'yolo.cfg.txt')

# initialize the list of class labels
classes = []
with open('names.txt', 'r') as f:
    classes = f.read().splitlines()

# load the video
cap = cv2.VideoCapture("CCTV Footage.mp4")


while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # create a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # pass the blob through the network and obtain detections
    network.setInput(blob)
    out_layers_names = network.getUnconnectedOutLayersNames()
    layerOut = network.forward(out_layers_names)

    # initialize the required arrays
    boxes = []
    confidences = []
    class_ids = []

    # loop over detections
    for out in layerOut:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # obtain the indexes of detected objects
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # declare the font and the colors of the text to be displayed on detected objects
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # create rectangle on detected objects and put label
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 20), font, 0.5, (255, 255, 255), 1)

    # display the output video
    cv2.imshow('Output Video', img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
