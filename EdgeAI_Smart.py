from tkinter import *

#from detection import fire_detection, suspect_localisation, action_recognition
import argparse
import sys
import time
import face
import cv2
import tensorflow.compat.v1 as tf  # compatibility mode of tf1 in tf2
from PIL import Image, ImageTk
import json
import os

global configuration
global homedir
homedir = os.getcwd()

if not os.path.exists("config.json"):
  configuration={}
  configuration['classNames']=['chapeau', 'echarpe', 'gants', 'cle', 'lunettes', 'parapluie', 'telephone', 'telecommande', 'portefeuille', 'fire', 'smoke']
  configuration['notificationClasses']=['telephone', 'fire', 'smoke']
  configuration['yolo']={'confidence_level':.5,'activated_classes':[0,1,2,3,4,5,6,7,8,9,10]}
  configuration['smsActivated']=0
  configuration['teamsActivated']=0
  configuration['smsConfig']={
      'to_numbers': ["+32465294650"]
  }
  configuration['teamsConfig']={
      'url': "https://alumniumonsac.webhook.office.com/webhookb2/079e6963-be2e-4a27-a841-2ca66992b92d@488bed9d-d6a7-48d5-ba1f-ebec3823b357/IncomingWebhook/5636c849f828489cb959dd90a84134a9/73aee495-f3db-45bc-877d-61d3ea62d7e4"
  }
  configuration['allowedUser']=["dominique", "olivier dm", "adam", "philippe", "hughes", "frank"]
  with open('config.json', 'w') as outfile:
      json.dump(configuration, outfile, indent=4)
else:
  with open('config.json') as json_file:
      configuration = json.load(json_file)

#" ".join([str(i) for i in configuration['yolo']['activated_classes']])

#physical_devices=tf.config.list_physical_devices('GPU') #tf.config.experimental.set_memory_growth(physical_devices[0], True)

#allowedUser = ["dominique", "olivier_dm", "adam", "philippe", "hughes", "frank"]
global ok
global loggedInUser
loggedInUser={}
for user in configuration['allowedUser']:
  loggedInUser[user]=0

#loggedInUser['olivier_dm']=10

def add_overlays(frame, faces, frame_rate, ok):
    names=[]
    if faces is not None:
        for face in faces:
            if face.name is not None:
                print(face.name)
                if face.name in configuration['allowedUser']:                                            # Green
                    loggedInUser[face.name]+=1
                    ok += 1
                    if (ok==20):
                        authenticatedPerson = max(loggedInUser, key=loggedInUser.get)
                        print ("Congratulations {}, you are well authentified".format(authenticatedPerson))
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

    while (ok < 20):
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

def save_config():
    global configuration
    configuration['smsActivated'] = var1.get()
    configuration['teamsActivated'] = var2.get()
    with open('config.json', 'w') as outfile:
      json.dump(configuration, outfile, indent=4)

def load_config():
    global configuration
    with open('config.json') as json_file:
      configuration = json.load(json_file)
    var1.set(configuration['smsActivated'])
    var2.set(configuration['teamsActivated'])
    yoloConfLabel.set("Confidence level: {}".format(configuration['yolo']['confidence_level']))
    yoloClassesLabel.set("Activated classes: {}".format(activatedClassesToString()))
    yoloNotificationLabel.set("Notifications: {}".format(", ".join(configuration['notificationClasses'])))



def activatedClassesToString():
    global configuration
    listOfActivatedClasses=", ".join([configuration['classNames'][i] for i in configuration['yolo']['activated_classes']])
    output_string=[]
    for i in range(0,len(listOfActivatedClasses),40):
      output_string.append(listOfActivatedClasses[:40])
      listOfActivatedClasses=listOfActivatedClasses[40:]
    return "\n".join(output_string)

def run_fire():
    os.chdir( "{}/../fire/".format(homedir) )
    os.system('python Video_detect_fire.py')


def run_yolo():
    os.chdir( "{}/Yolo_V5/yolov5/".format(homedir) )
    os.system('export SMS_SID=ACe399538ba69578fa49aec9e458e85863 export SMS_TOKEN=a068cff3aaa9a88ba98ad480b82f7831')
    os.system('python detect.py --weights ../../../Yolo_V5/yolov5wfire.pt --img 640 --conf {} --classes {} --source 0'.format(configuration['yolo']['confidence_level']," ".join([str(i) for i in configuration['yolo']['activated_classes']])))

def run_emotions():
    os.chdir( "{}/emotions".format(homedir) )
    os.system('python Face\ -\ Detect.py')

def run_movements():
    os.chdir( "{}/../MMAction/".format(homedir) )
    os.system('DBUS_FATAL_WARNINGS=0 ffplay BaillementChute.mp4')

def open_config():
    os.chdir( "{}/".format(homedir) )
    os.system('gedit config.json')


    
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
    gwidth=850
    gheight=550
    swidth=root.winfo_screenwidth()
    sheight=root.winfo_screenheight()
    root.geometry('{}x{}+{}+{}'.format(gwidth,gheight,int(swidth/2-gwidth/2),int(sheight/2-gheight/2)))
    root.title('Freakin Awesome Smart Home System')


    # create all of the main containers
    top_frame = Frame(root, width=gwidth-10, height=50, pady=3)
    center = Frame(root, width=gwidth-10, height=40, padx=3, pady=3)
    btm_frame = Frame(root, width=gwidth-10, height=45, pady=3)

    # layout all of the main containers
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    top_frame.grid(row=0, sticky="ew")
    center.grid(row=1, sticky="nsew")
    #btm_frame.grid(row=3, sticky="ew")

    # create the widgets for the top frame
    image_logo = Image.open("logo.png")
    logoHeight=int(image_logo.size[1]*(gwidth*0.2)/image_logo.size[0])
    image_logo = image_logo.resize((int(gwidth*0.2), logoHeight), Image.ANTIALIAS)
    photo_logo = ImageTk.PhotoImage(image_logo)
    img_label_logo = Label(top_frame,image=photo_logo)
    img_label_logo.image = photo_logo

    model_label = Label(top_frame, text="Welcome to your smart home \n {}!".format(max(loggedInUser, key=loggedInUser.get)), font=('Helvetica 24 bold'))
    
    image_id = Image.open("ID/{}.jpg".format(max(loggedInUser, key=loggedInUser.get).replace(" ","_")))
    image_id = image_id.resize((int(image_id.size[0]*(logoHeight)/image_id.size[1]), logoHeight), Image.ANTIALIAS)
    photo_id = ImageTk.PhotoImage(image_id)
    img_label_id = Label(top_frame,image=photo_id)
    img_label_id.image = photo_id
    

    #width_label = Label(top_frame, text='Width:')
    #length_label = Label(top_frame, text='Length:')
    #entry_W = Entry(top_frame, background="pink")
    #entry_L = Entry(top_frame, background="orange")

    # layout the widgets in the top frame
    top_frame.grid_rowconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=2)

    img_label_logo.grid(row=0,column=0,sticky="w")
    model_label.grid(row=0, column=1,sticky="ns")
    img_label_id.grid(row=0, column=2, sticky="e")
    #width_label.grid(row=1, column=0)
    #length_label.grid(row=1, column=2)
    #entry_W.grid(row=1, column=1)
    #entry_L.grid(row=1, column=3)

    # create the center widgets
    center.grid_rowconfigure(0, weight=1)
    center.grid_columnconfigure(1, weight=2)

    ctr_left = Frame(center, width=int(gwidth/2))
    #ctr_mid = Frame(center, bg='yellow', width=100, height=190, padx=3, pady=3)
    ctr_right = Frame(center, width=int(gwidth/2))

    ctr_left.grid_rowconfigure(0, weight=1)
    ctr_left.grid_columnconfigure(1, weight=1)
    ctr_right.grid_rowconfigure(0, weight=1)
    ctr_right.grid_columnconfigure(1, weight=1)

    label_l_title = Label(ctr_left, text="Detection tools", font=('Helvetica 20 bold'))
    label_l_title.grid(row=0,column=0,sticky="nw")
    but1 = Button(ctr_left,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=run_fire,#fire_detection,
                  text='Fire detection',
                  font=('helvetica 15 bold'))
    but2 = Button(ctr_left,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=run_yolo,#fire_detection,
                  text='Object+Fire detection',
                  font=('helvetica 15 bold'))
    global yoloClassesLabel,yoloConfLabel,yoloNotificationLabel
    yoloClassesLabel = StringVar()
    yoloNotificationLabel = StringVar()
    
    yoloConfLabel = StringVar()
    yoloConfLabel.set("Confidence level: {}".format(configuration['yolo']['confidence_level']))
    label_l_yolocl = Label(ctr_left, textvariable=yoloConfLabel, font=('Helvetica 12'))

    

    yoloClassesLabel.set("Activated classes: {}".format(activatedClassesToString()))  
    yoloNotificationLabel.set("Notifications: {}".format(", ".join(configuration['notificationClasses'])))
      
    label_l_yoloclasses = Label(ctr_left, textvariable=yoloClassesLabel, font=('Helvetica 12'))
    label_l_yolonotif = Label(ctr_left, textvariable=yoloNotificationLabel, font=('Helvetica 12'))
    
    but3 = Button(ctr_left,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=run_emotions,#fire_detection,
                  text='Show your emotions',
                  font=('helvetica 15 bold'))
    but3b = Button(ctr_left,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=run_movements,#fire_detection,
                  text='Detected movements',
                  font=('helvetica 15 bold'))

    but1.grid(row=1,column=0,sticky="nw")
    but2.grid(row=2,column=0,sticky="nw")
    label_l_yolocl.grid(row=3,column=0,sticky="nw")
    label_l_yoloclasses.grid(row=4,column=0,sticky="nw")
    label_l_yolonotif.grid(row=5,column=0,sticky="nw")
    but3.grid(row=6,column=0,sticky="nw")
    but3b.grid(row=7,column=0,sticky="nw")

    
    label_l_not = Label(ctr_right, text="Notifications", font=('Helvetica 20 bold'))
    label_l_not.grid(row=0,column=0,sticky="nw")
    label_l_sms = Label(ctr_right, text=" SMS", font=('Helvetica 15'))
    label_l_sms.grid(row=1,column=0,sticky="nw")
    global var1, var2
    var1 = IntVar()
    var1.set(configuration['smsActivated'])
    chkSMS = Checkbutton(ctr_right, variable=var1)
    chkSMS.grid(row=1,column=1, sticky="w")
    label_l_tms = Label(ctr_right, text=" TEAMS", font=('Helvetica 15'))
    label_l_tms.grid(row=2,column=0,sticky="nw")
    var2 = IntVar()
    var2.set(configuration['teamsActivated'])
    chkTMS = Checkbutton(ctr_right, variable=var2)
    chkTMS.grid(row=2,column=1, sticky="w")    

    but4 = Button(ctr_right,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=save_config,#fire_detection,
                  text='Save config',
                  font=('helvetica 15 bold'))
    but5 = Button(ctr_right,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=load_config,#fire_detection,
                  text='Reload config',
                  font=('helvetica 15 bold'))
    but5b = Button(ctr_right,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=open_config,#fire_detection,
                  text='Edit config',
                  font=('helvetica 15 bold'))
    but6 = Button(ctr_right,
                  padx=5, pady=5,
                  # bd=5,
                  width=39, bg='white', fg='black',
                  relief=RAISED,
                  command=root.destroy,#fire_detection,
                  text='Log out',
                  font=('helvetica 15 bold'))
    but4.grid(row=3,column=0,columnspan=2,sticky="nesw")
    but5.grid(row=4,column=0,columnspan=2,sticky="nesw")
    but5b.grid(row=5,column=0,columnspan=2,sticky="nesw")
    but6.grid(row=6,column=0,columnspan=2,sticky="nesw")




    ctr_left.grid(row=0, column=0, sticky="new")
    #ctr_mid.grid(row=0, column=1, sticky="nsew")
    ctr_right.grid(row=0, column=1, sticky="new")


    root.mainloop()


if __name__ == '__main__':
    main()
