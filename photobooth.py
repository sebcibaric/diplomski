import sys
import os 
import cv2
import time
import argparse
import numpy as np
import face_extractor as fe

parser = argparse.ArgumentParser(description="Takes batch of photos")
parser.add_argument('-n', '--name', type=str, nargs='+', help='Person\'s name, required argument', required=True)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

name = '_'.join(args.name)
purposes = ['train', 'valid']

for purpose in purposes:
    path = './dataset/{}/{}'.format(purpose, name)
    if not os.path.exists(path):
        print('[INFO] Path ' + path + ' does not exist. It will be created') 
        os.mkdir(path)

purpose = purposes[0]
lista = []
cam = cv2.VideoCapture(0)
i = 0
while i < 200:

    ret, frame = cam.read()

    if not ret:
        print('[ERROR] Failed to take a picture')
        break

    extracted_face = fe.face_extract(frame, 40)

    if type(extracted_face) is not np.ndarray:
        print('[INFO] Face not found')
        continue

    if i == 160:
        purpose = purposes[1]
    
    path = './dataset/{}/{}'.format(purpose, name)
    time_to_int = int(time.time())
    timestamp = str(time_to_int + i)

    img = '{}/{}.jpg'.format(path, timestamp)
    cv2.imwrite(img, extracted_face)
    print(img)
    i = i + 1

cam.release()
