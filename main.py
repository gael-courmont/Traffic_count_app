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
img_size=416
conf_thres=0.8
nms_thres=0.4
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')
def detect_image(img,model):
    with torch.no_grad():
        detections = model(img)
    return detections

def app(videoPath):
    car_number = 0
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    vehicle_crossing=[]
    vid = cv2.VideoCapture(videoPath)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,force_reload=True)
    while True:
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame,(500,500))
        ih, iw, channels = frame.shape

        ##draw boundaries

        cv2.line(frame, (0, middle_line_position), (iw, middle_line_position),
                 (255, 0, 255), 1)
        cv2.line(frame, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
        cv2.line(frame, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)



        if ret:
            pilimg = Image.fromarray(frame)
            img = np.array(pilimg)
            detections = detect_image(pilimg, model)
            #detections=cv2.dnn.NMSBoxes(detections.pandas().xyxy[0][['xmin','ymin','xmax','ymax']],detections.pandas().xyxy[0]['confidence'],conf_thres,nms_thres)
            pad_x = max(img.shape[0] - img.shape[1], 0) *(img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) *(img_size / max(img.shape))
            unpad_h = img_size - pad_y
            unpad_w = img_size - pad_x
            if detections is not None:
                tracked_objects = mot_tracker.update(detections.xyxy[0])
                print(tracked_objects)
                for x1, y1, x2, y2, index, label in tracked_objects:

                    color = colors[int(label) % len(colors)]
                    color = [i * 255 for i in color]
                    frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                        color, 4)
                    frame = cv2.putText(frame,f'{classNames[int(label)]} {int(index)}',(int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    if ( (float(y1)<middle_line_position) & (float(y1)>middle_line_position-2)  ) :
                        car_number=car_number+1

            frame = cv2.putText(frame, str(car_number), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,250,0), 1)
            cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Use a breakpoint in the code line below to debug your script.







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app('data/british_highway_traffic.mp4')
    #app('data/traffic_detection.mp4')
    #app('data/Pexels_Videos _034115.mp4')
    #app('data/Pexels Videos 1192116.mp4')


