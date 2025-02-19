import os
import pickle
import mediapipe as mp
import cv2
from tqdm import tqdm  # For progress bars

mp_hands = mp.solutions.hands
DATA_DIR = './data'

data = []
labels = []

# Get total number of images to process
total_images = sum(len(os.listdir(os.path.join(DATA_DIR, dir_))) 
                  for dir_ in os.listdir(DATA_DIR) 
                  if not dir_.startswith('.') and os.path.isdir(os.path.join(DATA_DIR, dir_)))

print(f"Found {total_images} images to process")

with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
    for dir_ in os.listdir(DATA_DIR):
        if dir_.startswith('.') or not os.path.isdir(os.path.join(DATA_DIR, dir_)):
            continue
            
        print(f"\nProcessing class: {dir_}")
        dir_path = os.path.join(DATA_DIR, dir_)
        
        for img_path in tqdm(os.listdir(dir_path), desc=f"Processing {dir_}"):
            if img_path.startswith('.'):
                continue
                
            try:
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(dir_path, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
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

                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"\nNo hand detected in {img_path}")
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")

print(f"\nProcessing complete!")
print(f"Total samples collected: {len(data)}")
print(f"Classes found: {set(labels)}")

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("✅ Data saved to 'data.pickle'")
