import os
import json
import cv2
import argparse
import numpy as np
import face_extractor as fe
import drivers
from keras.models import load_model
from time import sleep
from PIL import Image


def face_rec(args, model, labels):
    cam = cv2.VideoCapture(0)
    display = None

    if args.lcd:
        display = drivers.Lcd()

    try:
        while True:
            ret, frame = cam.read()

            input_im = fe.face_extract(frame, 0)

            if type(input_im) is not np.ndarray:
                print('[ERROR] No faces found')
                sleep(2)
                continue

            input_im = cv2.resize(input_im, (224, 224), interpolation=cv2.INTER_LINEAR)

            im = Image.fromarray(input_im, 'RGB')

            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            print(predictions)
            index = np.argmax(predictions[0])
            val = predictions[0][index]
            name = ''
            message = ''
            if val > 0.6:
                name = labels.get(index)
                message = 'osoba ispred kamere je {}'.format(name)
            else:
                message = 'osoba nije prepoznata'

            if args.lcd:
                display.lcd_display_string("Ispred kamere je", 1)
                display.lcd_display_string("{}".format(name), 2)
            elif args.cli:
                print(message)

            sleep(4)
    except KeyboardInterrupt:
        print('[WARN] Program shutdown')

    if args.lcd:
        display.lcd_clear()

    cam.release()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Izbjegava koristenje CUDA na PC-u
    parser = argparse.ArgumentParser(description="Recognising faces")
    parser.add_argument('--cli', action='store_true', help='Argument for CLI output only')
    parser.add_argument('--lcd', action='store_true', help='Argument for LCD output only')

    args = parser.parse_args()

    model = load_model('model.h5')

    labels = {}
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
        labels = {v: k for k, v in data.items()}

    print(labels)

    face_rec(args, model, labels)


if __name__ == '__main__':
    main()
