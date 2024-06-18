import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

face_classifier = cv2.CascadeClassifier('D:\Mus_Ap\emotionDetection\emotionDetection\haarcascade_frontalface_default.xml')
emotion_classifier = load_model('D:\Mus_Ap\emotionDetection\emotionDetection\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label, (x, y, w, h)
    return None, None

def main():
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_emotion, face_coords = detect_emotion(frame)
        
        if detected_emotion:
            if face_coords:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print("Detected Emotion:", detected_emotion)

        cv2.imshow('Emotion Detector', frame)


        if time.time() - start_time >= 5:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Final Emotion:", detected_emotion)

if __name__ == '__main__':
    main()
