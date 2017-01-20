import glob
from shutil import copyfile
import cv2

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("source_emotion/*") #Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    print(part)
    raw_input()
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        print(sessions)
        raw_input()
        for files in glob.glob("%s/*" %sessions):
            print(files)
            raw_input()
            current_session = files[20:-30]
            print(current_session)
            raw_input()
            file = open(files, 'r')
            print(file)
            raw_input()
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            print(emotion)
            raw_input()
            sourcefile_emotion = glob.glob("source_images/%s/%s/*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            print(sourcefile_emotion)
            raw_input()
            sourcefile_neutral = glob.glob("source_images/%s/%s/*" %(part, current_session))[0] #do same for neutral image
            print(sourcefile_neutral)
            raw_input()
            dest_neut = "sorted_set/neutral/%s" %sourcefile_neutral[25:] #Generate path to put neutral image
            dest_emot = "sorted_set/%s/%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image
            
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def detect_faces(emotion):
    files = glob.glob("sorted_set/%s/*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
            frame = frame[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(frame, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("dataset1/%s/%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion) #Call functiona
