import os
import uuid
import cv2
import numpy as np
import librosa
import tensorflow as tf
from collections import deque
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==== Load Models and Configs ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image model
IMG_MODEL = model_from_json(open(os.path.join(BASE_DIR, 'models/emotion_modelIV.json')).read())
IMG_MODEL.load_weights(os.path.join(BASE_DIR, 'models/modelIV.weights.h5'))
IMG_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Video model
VID_MODEL = model_from_json(open(os.path.join(BASE_DIR, 'models/emotion_modelIV.json')).read())
VID_MODEL.load_weights(os.path.join(BASE_DIR, 'models/modelIV.weights.h5'))
VID_LABELS = IMG_LABELS

# Audio model
AUDIO_MODEL = model_from_json(open(os.path.join(BASE_DIR, 'models/audio_emotion_model.json')).read())
AUDIO_MODEL.load_weights(os.path.join(BASE_DIR, 'models/audio_model.weights.h5'))
AUDIO_LABELS = [
    "neutral",  "calm",     "happy", "sad",
    "angry",    "fearful",  "disgust","surprised"

]

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 216

# Text model
TEXT_MODEL = model_from_json(open(os.path.join(BASE_DIR, 'models/text_model.json')).read())
TEXT_MODEL.load_weights(os.path.join(BASE_DIR, 'models/text_model.weights.h5'))
with open(os.path.join(BASE_DIR, 'models/tokenizer_config.json'), 'r') as f:
    tokenizer = tokenizer_from_json(f.read())
TEXT_LABELS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
MAX_SEQ_LEN = 50



# =================== Prediction Functions ===================

def predict_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return "No Face Detected", None

    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        prediction = IMG_MODEL.predict(face, verbose=0)
        emotion = IMG_LABELS[np.argmax(prediction)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    result_filename = f"{uuid.uuid4().hex}_result.jpg"
    result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
    cv2.imwrite(result_path, img)
    return emotion, result_filename

def predict_realtime_video():
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    smooth = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
            roi = np.expand_dims(roi, -1)
            roi = np.expand_dims(roi, 0)
            preds = VID_MODEL.predict(roi, verbose=0)
            label = int(np.argmax(preds))
            smooth.append(label)
            display_label = VID_LABELS[max(set(smooth), key=smooth.count)]
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.putText(frame, display_label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Real‑time Emotion", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def extract(file_path,max_pad_len=MAX_LEN):
    y,sr=librosa.load(file_path,sr=SAMPLE_RATE)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)
    pad=max_pad_len-mfcc.shape[-1]
    if pad>0:
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc=mfcc[:,:max_pad_len]
    return mfcc
def predict_audio(audio_path):
    mfcc = extract(audio_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    pred  = AUDIO_MODEL.predict(mfcc, verbose=0)[0]
    idx   = int(np.argmax(pred))
    print(pred)
    return AUDIO_LABELS[idx]


def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN)
    pred = TEXT_MODEL.predict(pad, verbose=0)[0]
    return TEXT_LABELS[np.argmax(pred)]

# =================== Views ===================

def home(request):
    return render(request, 'home.html')

def main(request):
    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        if form_type == 'image':
            file_obj = request.FILES.get("image")
            tmp_path = os.path.join(settings.MEDIA_ROOT, file_obj.name)

            with open(tmp_path, "wb+") as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)

            emotion, result_filename = predict_image(tmp_path)
            return render(request, "result.html", {
                "input_type": "image",
                "emotion": emotion,
                "image_url": settings.MEDIA_URL + result_filename
            })

        elif form_type == 'realtime_video':
            predict_realtime_video()
            return HttpResponse("Real-time video prediction finished.")

        elif form_type == 'audio' and 'audio' in request.FILES:
            audio = request.FILES['audio']
            path = os.path.join(settings.MEDIA_ROOT, audio.name)
            with open(path, 'wb+') as f:
                for chunk in audio.chunks():
                    f.write(chunk)
            request.session['emotion'] = predict_audio(path)
            request.session['input_type'] = 'audio'
            return redirect('result')

        elif form_type == 'text':
            text = request.POST.get('text')
            request.session['emotion'] = predict_text(text)
            request.session['input_type'] = 'text'
            return redirect('result')

    return render(request, 'main.html')

def result(request):
    input_type = request.GET.get("input_type") or request.session.get('input_type')
    emotion = request.GET.get("emotion") or request.session.get('emotion')
    image_url = request.GET.get("image_url")

    return render(request, 'result.html', {
        "input_type": input_type,
        "emotion": emotion,
        "image_url": image_url
    })
