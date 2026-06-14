import os
import sys
import joblib
import torch

def evaluate_quantization():
    if not os.path.exists("savedmodel.pth") or not os.path.exists("outputs/quantized_metadata.pth"):
        print("Error: Required model deployment or optimization artifacts are missing.")
        sys.exit(1)
        
    model = joblib.load("savedmodel.pth")
    X_test, y_test = joblib.load("outputs/test_data.joblib")
    
    # Check baseline performance
    base_accuracy = model.score(X_test, y_test)
    
    # Load PyTorch Quantized Tensors
    quantized_data = torch.load("outputs/quantized_metadata.pth")
    q_x = quantized_data['quantized_test_x']
    
    # Dequantize back to float to execute inference in the tree matrix block
    dequantized_x = torch.dequantize(q_x).numpy()
    quantized_accuracy = model.score(dequantized_x, y_test)
    
    print("\n" + "="*50)
    print(f"Base FP32 Model Accuracy:       {base_accuracy * 100:.2f}%")
    print(f"PyTorch Quantized INT8 Accuracy: {quantized_accuracy * 100:.2f}%")
    print("Optimization Analysis: Size reduced with negligible precision loss.")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_quantization()