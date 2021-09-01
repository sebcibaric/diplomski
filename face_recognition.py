import json
import sys
import cv2
import numpy as np
import face_extractor as fe
from keras.models import load_model
from PIL import Image

input_im = cv2.imread('./iva.jpg')

input_im = fe.face_extractor(input_im)

if type(input_im) is not np.ndarray:
    print('[ERROR] no faces found')
    sys.exit()

input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)

im = Image.fromarray(input_im, 'RGB')

img_array = np.array(im)
img_array = np.expand_dims(img_array, axis=0)

model = load_model('model.h5')

predictions = model.predict(img_array)
print(predictions)
id = np.argmax(predictions, axis=1)[0]
print(id)

name = ''
labels = {}
with open('data.json', 'r') as json_file:
    data = json.load(json_file)
    labels = { v:k for k, v in data.items() }
    name = labels.get(id)

print('osoba na slici je {}'.format(name))