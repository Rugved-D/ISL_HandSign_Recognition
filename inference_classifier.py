import pickle
import numpy as np
import mediapipe as mp
import cv2
import os

# Load the trained model
if not os.path.exists('model.p'):
    raise FileNotFoundError("❌ Model file 'model.p' not found. Train the model first.")

if not os.path.exists('label_encoder.pickle'):
    raise FileNotFoundError("❌ Label encoder file 'label_encoder.pickle' not found. Train the model first.")

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Ensure the feature length is correct
            if len(data_aux) < 84:
                data_aux += [0] * (84 - len(data_aux))
            elif len(data_aux) > 84:
                data_aux = data_aux[:84]

            prediction = model.predict([data_aux])[0]
            predicted_character = label_encoder.inverse_transform([prediction])[0]

            cv2.putText(frame, f'Prediction: {predicted_character}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Sign Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
