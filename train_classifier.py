import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the data
print("📂 Loading dataset...")
data_dict = pickle.load(open('data.pickle', 'rb'))

# Ensure consistent feature length (42 landmarks * 2 = 84)
expected_length = 84
cleaned_data = []
cleaned_labels = []

for i, (d, label) in enumerate(zip(data_dict['data'], data_dict['labels'])):
    if len(d) > 0:  
        if len(d) > expected_length:
            cleaned_data.append(d[:expected_length])  # Truncate if longer
        else:
            cleaned_data.append(d + [0] * (expected_length - len(d)))  # Pad if shorter
        cleaned_labels.append(label)

data = np.array(cleaned_data)
labels = np.array(cleaned_labels)

# Encode labels (A -> 0, B -> 1, ..., Z -> 25, 0 -> 26, 1 -> 27, ...)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved as 'label_encoder.pickle'")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Train the model
print("🛠️ Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'\n🎯 Model Accuracy: {accuracy * 100:.2f}%')
print('\n📊 Classification Report:')
print(classification_report(y_test, y_predict))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("✅ Model saved as 'model.p'")
