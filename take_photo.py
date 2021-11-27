import sys
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(description="Takes one photo")
    parser.add_argument('-n', '--name', type=str, nargs='+', help='Person\'s name, required argument', required=True)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    if not ret:
        print('[ERROR] Failed to take a picture')
        sys.exit()

    name = '_'.join(args.name)
    img = '{}.jpg'.format(name)
    cv2.imwrite(img, frame)
    cam.release()
    print(img)


if __name__ == '__main__':
    main()

