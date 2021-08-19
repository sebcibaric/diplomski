import sys
import cv2

name = sys.argv[1]

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

if not ret:
    print('[ERROR] Failed to take a picture')
    sys.exit()

img = '{}.jpg'.format(name)
cv2.imwrite(img, frame)
print(img)
