import keras
import cv2
import numpy as np

from mtcnn import MTCNN
from PIL import Image

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# detect and return face locations
def detect(frame, detector, emotion_model):
    # convert to RGB
    rgb_frame = frame[:, :, ::-1]

    results = []
    detector_res = detector.detect_faces(rgb_frame)
    for face in detector_res:
        # 1. get coordinates of faces
        coordinates, max_border = detect_box_coordinates(face)
        # 2. crop face to detect
        x = crop_to_predict(rgb_frame, coordinates, max_border, [48, 48], (1, 48, 48, 1), True)
        # 3. detect expession
        predictions = emotion_model.predict(x)
        emotion_label = emotion_labels[np.argmax(predictions)]
        results.append([coordinates, emotion_label])

    return results


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def detect_box_coordinates(face):
    x, y, width, height = face['box']
    center = [x + (width / 2), y + (height / 2)]
    max_border = max(width, height)
    top = max(int(center[1] - (max_border / 2)), 0)
    right = max(int(center[0] + (max_border / 2)), 0)
    bottom = max(int(center[1] + (max_border / 2)), 0)
    left = max(int(center[0] - (max_border / 2)), 0)
    return [top, right, bottom, left], max_border


def crop_to_predict(img, coordinates, max_border, size, shape, to_gray):
    top = coordinates[0]
    left = coordinates[3]
    x = img[top:top + max_border, left:left + max_border, :]
    x = np.array(Image.fromarray(x).resize(size))
    if to_gray:
        x = rgb2gray(x)
    return x.reshape(shape)


# ****************************************************************************************************
# ------------------------------------------------------------
# Main line
# ------------------------------------------------------------
if __name__ == "__main__":
    detector = MTCNN()
    emotion_model = keras.models.load_model("model.h5")
    detect_in_progress = False

    video_capture = cv2.VideoCapture(0)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not detect_in_progress:
            detect_in_progress = True
            results = detect(frame, detector, emotion_model)
            for location, emotion_label in results:
                top, right, bottom, left = location
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # put text
                cv2.putText(frame, emotion_label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            detect_in_progress = False

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
