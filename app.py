from flask import Flask, render_template, request ,redirect
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import fnmatch
import os
import random
import time




UPLOAD_FOLDER = './static/'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
print("+"*50, "Loading Model File For Analyze........ ")
classifier = load_model('./model.h5')
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/emotion_detect', methods=['POST'])
def emotion_detect():
    type= request.form['type']
    found = False
    moods = []
    start = time.time()
    cap = cv2.VideoCapture(0)
    while not found:
        
        
        ret, frame = cap.read()
        labels = []
        end = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi_gray)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
           
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                
                print("\nprediction = ",end - start)
                print("\nprediction = ", preds)
                print("\nlabel = ", label)
                print("\nprediction max = ", preds.argmax())
                moods.append(label)
                print("\nprediction moods = ", moods)
                if len(moods)> 7:
                    found = True
                    label = max(moods, key = moods.count)
                    if label:
                        cap.release()
                        song1 = random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/" +type+ "/"))
                        if label == 'Neutral' or label == 'Happy':
                            popup = True
                        else:
                            popup = False

                        return render_template("emotion_detect.html", data=label,type=type, song=song1,popup=popup)
                    else:
                        cap.release()
                        return render_template('index.html',data='No Face Found')
                
            else:
                cap.release()
                return render_template('index.html',data='No Face Found')
        if (end-start)>15:
                    cap.release()
                    return render_template('index.html',data='No Face Found')

if __name__ == "__main__":
    app.run(debug=True)
