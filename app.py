from flask import Flask, render_template, request 
import numpy as np
import cv2
from keras.models import load_model
import os
import random
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/songs/'
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

@app.route('/',methods=['GET','POST'])
def uploadfile():
    mood= request.form['mood']
    lang= request.form['lang']
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], lang+'/'+mood+'/'+filename))
    return render_template('index.html')

@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    language= request.form['language']
    print(language)
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
    print(prediction)
    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()

    song1= random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/"+language+"/"+prediction+"/"))
    song2= random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/"+language+"/"+prediction+"/"))
    song3= random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/"+language+"/"+prediction+"/"))
    song4= random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/"+language+"/"+prediction+"/"))
    song5= random.choice(os.listdir("F:/gg/ppp/emotion-based-music-ai-main/static/songs/"+language+"/"+prediction+"/"))
    print('You are '+prediction+' !!!! please calm down:) ,I will play song for you :' + song1)
    
    return render_template("emotion_detect.html", data=prediction,lang=language, s1=song1, s2=song2, s3=song3, s4=song4, s5=song5)


if __name__ == "__main__":
    app.run(debug=True)
