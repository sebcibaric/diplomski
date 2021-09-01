import sys
import os 
import cv2
import time
import argparse

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
for i in range(0, 200):
    if i == 160:
        purpose = purposes[1]
    
    path = './dataset/{}/{}'.format(purpose, name)
    time_to_int = int(time.time())
    timestamp = str(time_to_int + i)

    ret, frame = cam.read()

    if not ret:
        print('[ERROR] Failed to take a picture')
        break

    img = '{}/{}.jpg'.format(path, timestamp)
    cv2.imwrite(img, frame)
    print(img)

cam.release()