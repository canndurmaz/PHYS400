import tensorflow as tf
import numpy as np
import json
import os
import sys

# Must match the training order in nn_alloy.py
ALL_ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Ni", "Ti", "Zn"]

def predict_from_json(config_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "alloy_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'src/ML/run_nn.sh' first to train and save the model.")
        return

    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and parse the composition
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path) as f:
        cfg = json.load(f)
        # Handle both lmp-style and simple-dict style
        comp = cfg.get("composition", cfg) 
        
    # Prepare feature vector
    features = np.zeros((1, len(ALL_ELEMENTS)))
    for el, frac in comp.items():
        if el in ALL_ELEMENTS:
            features[0, ALL_ELEMENTS.index(el)] = frac
        else:
            print(f"Warning: Element {el} not supported by the model.")

    # Generate prediction
    prediction_scaled = model.predict(features, verbose=0)
    
    # Unscale (E was divided by 100 during training)
    youngs_e = prediction_scaled[0, 0] * 100.0
    poisson_nu = prediction_scaled[0, 1]
    
    print("\n" + "="*50)
    print(f"{'Prediction for ' + os.path.basename(config_path):^50}")
    print("="*50)
    print(f"  Young's Modulus (E): {youngs_e:8.2f} GPa")
    print(f"  Poisson's Ratio (nu): {poisson_nu:8.3f}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_from_model.py <path_to_composition.json>")
    else:
        predict_from_json(sys.argv[1])
