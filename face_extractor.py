import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img, offset):
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)

    if not len(faces):
        return None

    for (x, y, w, h) in faces:
        x = x - offset
        y = y - offset
        cropped_face = img[y:y + h + offset, x:x + w + offset]

    return cropped_face