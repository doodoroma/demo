from tkinter import *

from detection import fire_detection, suspect_localisation, action_recognition
import argparse
import sys
import time
import face
import cv2
import tensorflow.compat.v1 as tf  # compatibility mode of tf1 in tf2

#physical_devices=tf.config.list_physical_devices('GPU') #tf.config.experimental.set_memory_growth(physical_devices[0], True)

allowedUser = ["dominique", "olivier_dm", "adam", "philippe", "hughes", "frank"]
global ok


def add_overlays(frame, faces, frame_rate, ok):
    names=[]
    if faces is not None:
        for face in faces:
            if face.name is not None:
                if face.name in allowedUser:                                            # Green
                    ok += 1
                    if (ok==50):
                        print ("Congratulations, you are well authentified")
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    return ok

def authentification():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    face_count=0
    ok = 0
    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    while (ok < 50):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        ok=add_overlays(frame, faces, frame_rate, ok)
        print(ok)
        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return ok


def main():
    tf.disable_v2_behavior()  # compatibility mode of tf1 in tf2
    ok=0
    # allow gpu grow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    ok=authentification()
    print ("ok:", ok)
    
    # create interface
    root = Tk()
    root.geometry('850x500')
    root.title('Edge AI System')

    # Principal frame
    main_frame = Frame(root, relief=RIDGE, borderwidth=2)
    main_frame.config(background='green1')
    main_frame.pack(fill=BOTH, expand=1)

    # Welcome message for user
    label_msg = Label(main_frame, text=("Welcome !"),
                      bg='green2', font=('Helvetica 24 bold'), height=2)
    label_msg.pack(side=TOP)
    label_msg2 = Label(main_frame, text=("Hello, you are well authorized, congrats "),
                       bg='green2', font=('Helvetica 22 bold'))
    label_msg2.pack(side=TOP)

    # Menu
    but1 = Button(main_frame,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=fire_detection,
                  text='Fire detection',
                  font=('helvetica 15 bold'))
    but1.place(x=200, y=150)
    but2 = Button(main_frame,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=suspect_localisation,
                  text='Suspect localisation',
                  font=('helvetica 15 bold'))
    but2.place(x=200, y=250)

    but3 = Button(main_frame,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=action_recognition,
                  text='Actions recognition',
                  font=('helvetica 15 bold'))
    but3.place(x=200, y=350)

    but4 = Button(main_frame,
                  padx=5, pady=5,
                  # bd=5,
                  width=12, bg='white', fg='black',
                  relief=RAISED,
                  command=root.destroy,
                  text='Exit',
                  font=('helvetica 15 bold'))
    but4.place(x=670, y=440)


    root.mainloop()


if __name__ == '__main__':
    main()
