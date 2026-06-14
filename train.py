import os
import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_pipeline():
    print("Fetching Olivetti faces dataset...")
    # fetch_olivetti_faces returns an object with 'data' and 'target' keys
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    
    print("Splitting dataset (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save test data alongside for validation testing in test.py
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "savedmodel.pth")
    joblib.dump((X_test, y_test), "outputs/test_data.joblib")
    print("Model successfully saved to savedmodel.pth")

if __name__ == "__main__":
    train_pipeline()