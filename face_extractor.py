import cv2


def face_extract(img):
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)

    return faces


def crop_face(frame, x, y, w, h):
    return frame[y:y + h, x:x + w]
