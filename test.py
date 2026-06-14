import os
import sys
import joblib

def evaluate_model():
    if not os.path.exists("savedmodel.pth"):
        print("Error: savedmodel.pth not found!")
        sys.exit(1)
        
    print("Loading model and test evaluation split...")
    model = joblib.load("savedmodel.pth")
    X_test, y_test = joblib.load("outputs/test_data.joblib")
    
    accuracy = model.score(X_test, y_test)
    print("\n" + "="*40)
    print(f"Decision Tree Test Set Accuracy: {accuracy * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()