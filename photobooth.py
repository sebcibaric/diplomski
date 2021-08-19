import sys
import os 
import cv2
import time

directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

arg_len = len(sys.argv)

if arg_len < 4:
    print('[ERROR] Missing arguments! '
          'Type in arguments for first name, last name purpose like: python3 photobooth.py john doe train/valid')
    sys.exit()
elif arg_len > 4:
    print('[ERROR] Too many arguments! Type in arguments '
          'for first name, last name and purpose like: python3 photobooth.py john doe train/valid')
    sys.exit()
elif arg_len == 4:
    name = sys.argv[1] + '_' + sys.argv[2] 
    purpose = sys.argv[3]

path = './dataset/{}/{}'.format(purpose, name)

if not os.path.exists(path):
    print('[INFO] Path ' + path + ' does not exist. It will be created') 
    os.mkdir(path)


cam = cv2.VideoCapture(0)
direction_counter = 0
for i in range(0, 40):
    time_to_int = int(time.time())
    timestamp = str(time_to_int + i)

    if (i == 0 or i % 10 == 0) and direction_counter < 4:
        print('Move your head: ' + directions[direction_counter])
        direction_counter = direction_counter + 1
        time.sleep(3)

    ret, frame = cam.read()

    if not ret:
        print('[ERROR] Failed to take a picture')
        break

    img = '{}/{}.jpg'.format(path, timestamp)
    cv2.imwrite(img, frame)
    print(img)
