import os
import glob
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.keras as keras

def test_raw_model(model_path):
    print("=" * 60)
    print(f"Testing model: {model_path}")
    try:
        # Load the raw keras model
        model = keras.models.load_model(model_path, compile=False)
        
        in_shape = model.input_shape
        print(f"Expected Input Shape: {in_shape}")
        
        if len(in_shape) != 2:
            print("Model expects a 3D+ input (e.g. RNN/LSTM). Cannot do simple 2D vector test.")
            return

        num_features = in_shape[1]
        
        # 1. Test with Raw Physical Values
        # Order is unknown (since it's baked in the wrapper), but let's 
        # test a typical distribution like [mDot, load, T] or [T, mDot, load].
        test_physical = np.zeros((1, num_features))
        if num_features == 3:
            # Let's assume order from training_nn [mDot, load, T] or similar
            test_physical[0] = [0.02, 150.0, 298.15]
            print(f"\n[Test 1] Feeding PHYSICAL inputs (e.g. [mDot: 0.02, load: 150, T: 298.15]):")
            print(f"Input: {test_physical[0]}")
        else:
            test_physical = np.ones((1, num_features))
            print(f"\n[Test 1] Feeding array of ONEs (unknown feature count = {num_features}):")

        out_phys = model.predict(test_physical, verbose=0)
        print(f"-> Prediction: {out_phys[0]}")

        # 2. Test with Normalized Values
        # If the network was trained with scaled data (0 to 1 or -1 to 1), 
        # feeding it physical data will blow it up.
        test_norm = np.zeros((1, num_features))
        if num_features == 3:
            test_norm[0] = [0.5, 0.5, 0.5]
            print(f"\n[Test 2] Feeding NORMALIZED inputs (e.g. 0.5 for all features):")
            print(f"Input: {test_norm[0]}")
        else:
            test_norm = np.zeros((1, num_features))
            print(f"\n[Test 2] Feeding array of ZEROs:")

        out_norm = model.predict(test_norm, verbose=0)
        print(f"-> Prediction: {out_norm[0]}")
        
        print("\nModel Architecture:")
        model.summary()
        print("\n")

    except Exception as e:
        print(f"Failed to load or test model: {e}")

if __name__ == "__main__":
    keras_file = os.path.join(
        os.path.dirname(__file__), "keras", "ml_model_fixed_norm.keras"
    )

    if not os.path.exists(keras_file):
        print("Specified .keras file not found.")
    else:
        test_raw_model(keras_file)
