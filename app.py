from flask import Flask, render_template, request ,redirect
import numpy as np
import cv2
from keras.models import load_model
import fnmatch
import os
import random


UPLOAD_FOLDER = './static/'
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+"*50, "Loading Model File For Analyze........ ")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)


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
    cap = cv2.VideoCapture(0)
    while not found:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)
        for x, y, w, h in faces:
            found = True
            roi = gray[y:y+h, x:x+w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))
    roi = roi/255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    prediction = model.predict(roi)
    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()
    i=1
    if prediction == 'Neutral' or prediction == 'Happy':
        print(prediction)
        return render_template('index.html',data='You Are In Neutral Mood')
        
    else:
        song1 = random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/" +type+ "/"))
        return render_template("emotion_detect.html", data=prediction,type=type, song=song1)
        

if __name__ == "__main__":
    app.run(debug=True)
