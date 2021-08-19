import json
import cv2
import numpy as np
from keras.models import load_model

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_classifier.detectMultiScale(frame_gray)

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face

input_im = cv2.imread('./iva.jpg')

input_im = face_extractor(input_im)

input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)
# input_im = cv2.resize(input_im, (224, 224))
input_im = input_im / 255.
input_im = input_im.reshape(1, 224, 224, 3)

model = load_model('model.h5')

predictions = model.predict(input_im)
print(predictions)
id = np.argmax(predictions, axis=1)[0]
print(id)

name = ''
with open('data.json', 'r') as json_file:
    data = json.load(json_file)
    for d in data:
        if id == data[d]:
            name = d

print('osoba na slici je {}'.format(name))