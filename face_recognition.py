import json
import sys
import cv2
import argparse
import numpy as np
import face_extractor as fe
from keras.models import load_model
from PIL import Image

def face_rec_cli(model, labels):
    cam = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cam.read()

            input_im = fe.face_extract(frame, 0)

            if type(input_im) is not np.ndarray:
                print('[ERROR] no faces found')
                continue

            input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)

            im = Image.fromarray(input_im, 'RGB')

            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            print(predictions)
            predictions_arr = predictions[0]
            
            for index, val in enumerate(predictions_arr):
                if val > 0.6:
                    name = labels.get(index)
                    print('osoba na slici je {}'.format(name))
                else:
                    print('osoba nije prepoznata')

    except KeyboardInterrupt:
        print('[WARN] Program shutdown')
    
    cam.release()


def face_rec_gui():
    print('napravi nesto drugo')

parser = argparse.ArgumentParser(description="Recognasing faces")
parser.add_argument('--no-vid', action='store_true', help='Argument for CLI output only')

args = parser.parse_args()

model = load_model('model.h5')

labels = {}
with open('data.json', 'r') as json_file:
    data = json.load(json_file)
    labels = { v:k for k, v in data.items() }

print(labels)
if args.no_vid:
    face_rec_cli(model, labels)
else:
    face_rec_gui()
