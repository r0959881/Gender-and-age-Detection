import cv2
import numpy as np
import tensorflow as tf

# Load Keras models
age_model = tf.keras.models.load_model('age_model.h5')
gender_model = tf.keras.models.load_model('gender_model.h5')

# Update these with your actual class order from train_data.class_indices!
# Example: If print(train_data.class_indices) gives {'11-15': 0, '16-20': 1, '21-25': 2, ...}
# then use: age_labels = ['11-15', '16-20', '21-25', ...]
age_labels = [
    '0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60', '61-70', '71-100'
]
gender_labels = ['female', 'male']  # Confirmed by your training mapping

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        # Preprocess for Keras model
        face_resized = cv2.resize(face_img, (128, 128))
        face_array = face_resized / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Predict gender
        gender_pred = gender_model.predict(face_array)
        gender = gender_labels[np.argmax(gender_pred)]

        # Predict age
        age_pred = age_model.predict(face_array)
        age = age_labels[np.argmax(age_pred)]

        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Keras Gender and Age Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()