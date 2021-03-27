import cv2
import time
import numpy as np
import os
from tkinter import *
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input #224*224
from keras.applications.vgg19 import VGG19, preprocess_input #224*224
from keras.applications.inception_v3 import InceptionV3,  preprocess_input #224*224
from keras.applications.inception_resnet_v2 import InceptionResNetV2,  preprocess_input #224*224
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201,  preprocess_input #224*224
from keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
from yolo import YOLO
font = cv2.FONT_HERSHEY_DUPLEX
import tensorflow as tf

def fire_detection():
    model_path="models/fire/MobileNet_sidi.h5"
    video_path="videos_benchmarks/fire/feu_test.avi"
    print(f"Running video {video_path} for model {model_path}")
    # load class names
    classes_path_fire = "labels/classes.txt"
    with open(classes_path_fire, 'r') as f:
        classes_fire = list(map(lambda x: x.strip(), f.readlines())) 
    # load model
    model_fire = tf.keras.models.load_model(model_path)
    # load video
    #capture = cv2.VideoCapture(video_path)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    top_n=1
    while True:
        ret, frame = capture.read()
        if ret == True:
            frame_rsz = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            x = image.img_to_array(frame_rsz)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            pred = model_fire.predict(x)[0]           			
            result = [(classes_fire[i], float(pred[i]) * 100.0) for i in range(len(pred))]
            result.sort(reverse=True, key=lambda x: x[1])
			
            for i in range(top_n):
                (class_name, prob) = result[i]
            cv2.putText(frame, class_name, (100, 100), font, 3, (255,0,0), 6, cv2.LINE_AA)
            cv2.imshow('Fire detection',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press Q on keyboard to  exit
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()

def suspect_localisation():
    # path to classes names
    model_path="models/objects/Floydhub/yolov3-sidi-9classes.h5"     
    video_path="videos_benchmarks/objects/suspect_test.avi"
    print(f"Running video {video_path} for model {model_path}")
    classes_suspect = "labels/9classes_suspect.txt"

    # load anchor
    if "tiny" in model_path:
        yolo_anchors_path = "labels/tiny_yolo_anchors.txt"
    else:
        yolo_anchors_path = "labels/yolo_anchors.txt"

    # load model
    model_yolo = YOLO(class_path1=classes_suspect, anchros_path1=yolo_anchors_path, model_path1=model_path)

    capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fps_display_interval = 1  # seconds
    frame_rate = 0
    frame_rates = []
    frame_count = 0
    start_time = time.time()
    elapsed_time = 0

    while True:
        ret, frame = capture.read()

        if ret == True:
            frame_rsz = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            yoloImage = Image.fromarray(frame_rsz)
            r_frame = model_yolo.detect_image(yoloImage)

            end_time = time.time()

            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                elapsed_time += end_time - start_time
                frame_count = 0
                frame_rates.append(frame_rate)
                start_time = time.time()

            frame_count += 1

            result = np.asarray(r_frame)

            cv2.putText(result, str(frame_rate) + " fps", (500, 50),
                        font, 1, (0, 255, 0), thickness=2, lineType=2)

            cv2.putText(result, 'Suspect object detection', (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Suspect object detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    
def action_recognition():
    # to be completed ......
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    top_n=1
    while True:
        ret, frame = capture.read()
        if ret == True:
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            # to be completed .....
            # to be completed .....
            cv2.putText(frame, 'Actions recognition', (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Actions recognition',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press Q on keyboard to  exit
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
