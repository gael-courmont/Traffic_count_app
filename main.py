# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sort import *
from sort.sort import *

mot_tracker = Sort()


middle_line_position = 235
up_line_position = middle_line_position - 30
down_line_position = middle_line_position + 30
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

keeping_label = [2, 5, 7, 3]


def detect_image(img, model):
    with torch.no_grad():
        detections = model(img)
    return detections


def count_vehicles(ypoint, xpoint, id, label):
    if (ypoint > up_line_position) and (ypoint < middle_line_position):
        if id not in temp_up_list:
            temp_up_list.append(id)
    elif (ypoint < down_line_position) and (ypoint > middle_line_position):
        if id not in temp_down_list:
            temp_down_list.append(id)
    elif ypoint < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[label] = up_list[label] + 1
    elif ypoint > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[label] = down_list[label] + 1


def calculate_box_center(x1, y1, x2, y2):
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2
    return x3, y3


def app(videoPath):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    vid = cv2.VideoCapture(videoPath)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    res = (1000, 500)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # codec
    out = cv2.VideoWriter('video.mp4', fourcc, 20.0, res)
    while True:
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1000, 500))
        ih, iw, channels = frame.shape
        ##draw boundaries
        if ret:
            pilimg = Image.fromarray(frame)
            img = np.array(pilimg)
            detections = detect_image(pilimg, model)
            cv2.line(frame, (0, middle_line_position), (iw, middle_line_position),
                     (255, 0, 255), 1)
            cv2.line(frame, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
            cv2.line(frame, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)
            if detections is not None:
                tracked_objects = mot_tracker.update(detections.xyxy[0].cpu())
                for x1, y1, x2, y2, index, label in tracked_objects:
                    color = colors[int(label) % len(colors)]
                    color = [i * 255 for i in color]
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          color, 4)
                    frame = cv2.putText(frame, f'{classNames[int(label)]} {int(index)}', (int(x1), int(y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    if label in keeping_label:
                        x3,y3=calculate_box_center(x1,y1,x2,y2)
                        count_vehicles(y3,x3, index, keeping_label.index(label))
            frame = cv2.putText(frame, "car going up: " + str(up_list[0]), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "car going down: " + str(down_list[0]), (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "truck going up: " + str(up_list[2]), (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "truck going down: " + str(down_list[2]), (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "Bus going up: " + str(up_list[1]), (250, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "Bus going down: " + str(down_list[1]), (250, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "Motorbike going up: " + str(up_list[3]), (250, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            frame = cv2.putText(frame, "Motorbike going down: " + str(down_list[3]), (250, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
            cv2.imshow('frame', frame)
            out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    out.release()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app('data/british_highway_traffic.mp4')

