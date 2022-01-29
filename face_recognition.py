import os
import json
import cv2
import argparse
import numpy as np
import face_extractor as fe
from tensorflow.keras.models import load_model
from time import sleep
from PIL import Image


def face_rec(args, model, labels):
    cam = cv2.VideoCapture(0)
    display = None

    if args.lcd:
        import drivers
        display = drivers.Lcd()

    try:
        while True:
            _, frame = cam.read()

            faces = fe.face_extract(frame)

            if not len(faces):
                print('[ERROR] No faces found')
                sleep(2)
                continue

            x, y, w, h = faces[0]
            name, message = recognize_face(frame, x, y, w, h, model, labels)

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


def face_rec_video(model, labels):
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()

        faces = fe.face_extract(frame)

        if type(faces) is not np.ndarray:
            cv2.putText(frame,"Face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        for (x, y, w, h) in faces:
            name, message = recognize_face(frame, x, y, w, h, model, labels)
            print(message)
            x1 = x
            y1 = y
            x2 = x1 + w
            y2 = y1 + h
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()


def recognize_face(frame, x, y, w, h, model, labels):
    input_im = fe.crop_face(frame, x, y, w, h)
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
    
    return name, message 


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Izbjegava koristenje CUDA na PC-u
    parser = argparse.ArgumentParser(description="Recognising faces")
    parser.add_argument('--cli', action='store_true', help='Argument for CLI output only')
    parser.add_argument('--lcd', action='store_true', help='Argument for LCD output only')
    parser.add_argument('--video', action='store_true', help='Argument for realtime video output only')

    args = parser.parse_args()

    model = load_model('model.h5')

    labels = {}
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
        labels = {v: k for k, v in data.items()}

    print(labels)

    if args.video:
        face_rec_video(model, labels)
    else:
        face_rec(args, model, labels)


if __name__ == '__main__':
    main()
