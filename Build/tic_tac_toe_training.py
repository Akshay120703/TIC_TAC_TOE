# tic_tac_toe_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset from CSV file
df = pd.read_csv('tic-tac-toe.csv')

# Encode the features
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split the dataset
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and LabelEncoder
joblib.dump(model, 'tic_tac_toe_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
