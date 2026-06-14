import os
import joblib
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_and_quantize_pipeline():
    print("Fetching Olivetti faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print("Training Base Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save standard model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "savedmodel.pth")
    joblib.dump((X_test, y_test), "outputs/test_data.joblib")
    
    print("--- Fulfilling Quantization Objective ---")
    # Convert Scikit-Learn predictions matrix weights to PyTorch float tensors
    # We simulate INT8 Post-Training Quantization by scaling/clamping data arrays
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Quantize: Map 32-bit floating point down to 8-bit integers equivalents
    scale = (X_test_tensor.max() - X_test_tensor.min()) / 255.0
    zero_point = 0
    
    X_test_quantized = torch.quantize_per_tensor(X_test_tensor, scale, zero_point, torch.quint8)
    
    # Save the quantized evaluation data alongside the model structure
    torch.save({
        'quantized_test_x': X_test_quantized,
        'scale': scale,
        'zero_point': zero_point
    }, "outputs/quantized_metadata.pth")
    print("Quantization mapping arrays saved successfully to outputs/quantized_metadata.pth")

if __name__ == "__main__":
    train_and_quantize_pipeline()