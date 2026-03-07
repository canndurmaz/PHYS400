import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os

# Elements we have across all potentials (10 unique elements)
ALL_ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Ni", "Ti", "Zn"]

def load_data():
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path) as f:
        results = json.load(f)
        
    X = []
    y = []
    
    for name, data in results.items():
        if "composition" in data:
            comp = data["composition"]
            # Create feature vector [Al, Co, Cr, Cu, Fe, Mg, Mn, Ni, Ti, Zn]
            features = [comp.get(el, 0.0) for el in ALL_ELEMENTS]
            X.append(features)
            # Create target vector [E_GPa, nu]
            target = [data["E_GPa"], data["nu"]]
            y.append(target)
            
    return np.array(X), np.array(y)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(len(ALL_ELEMENTS),)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(2)
    ])
    
    # Use a specific learning rate for better convergence on small data
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss='mse', metrics=['mae'])
    return model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data()
    if len(X) == 0:
        print("No data found in results.json. Please run simulations first.")
        return

    # Scale targets: E is ~70, nu is ~0.3
    y_scaled = y.copy()
    y_scaled[:, 0] /= 100.0
    
    model = build_model()
    
    print(f"\nTraining neural network on {len(X)} samples...")
    model.fit(X, y_scaled, epochs=1000, verbose=0)
    
    # Save the model
    model_path = os.path.join(script_dir, "alloy_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # 1. Training Set Performance (Verification)
    print("\n" + "="*70)
    print(f"{'Verification: Training Set Performance':^70}")
    print("="*70)
    print(f"{'Alloy Config':<25} | {'Truth (E/nu)':^18} | {'Pred (E/nu)':^18}")
    print("-"*70)
    
    train_preds_scaled = model.predict(X, verbose=0)
    for i in range(len(X)):
        e_truth, nu_truth = y[i]
        e_pred = train_preds_scaled[i, 0] * 100.0
        nu_pred = train_preds_scaled[i, 1]
        
        # We don't have the original filename easily here, so we'll just show indices or re-read
        print(f"Sample {i+1:<20} | {e_truth:6.1f} / {nu_truth:.3f} | {e_pred:6.1f} / {nu_pred:.3f}")

    # 2. Prediction for all entries in src/ML/predict.json
    predict_file = os.path.join(script_dir, "predict.json")
    if os.path.exists(predict_file):
        print("\n" + "="*70)
        print(f"{'Predictions for src/ML/predict.json':^70}")
        print("="*70)
        print(f"{'Alloy Name':<25} | {'E (GPa)':>8} | {'nu':>6}")
        print("-"*70)
        
        try:
            with open(predict_file) as f:
                predict_data = json.load(f)
            
            for alloy_name, comp in predict_data.items():
                new_alloy = np.zeros((1, len(ALL_ELEMENTS)))
                for el, frac in comp.items():
                    if el in ALL_ELEMENTS:
                        new_alloy[0, ALL_ELEMENTS.index(el)] = frac
                
                prediction_scaled = model.predict(new_alloy, verbose=0)
                e_pred = prediction_scaled[0, 0] * 100.0
                nu_pred = prediction_scaled[0, 1]
                print(f"{alloy_name:<25} | {e_pred:8.2f} | {nu_pred:6.3f}")
        except Exception as e:
            print(f"Error reading {predict_file}: {e}")
        print("="*70)

if __name__ == "__main__":
    main()
