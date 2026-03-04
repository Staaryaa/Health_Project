import numpy as np
import pandas as pd
import keras
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
# --- NEW IMPORT ---
from model import pred_model 

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH / "MODEL"
FINAL_MODEL_PATH = MODEL_PATH / "health_model.keras"

# --- MANUAL MODEL LOADING ---
def load_manual_model():
    # Build the skeleton using your model.py function
    # input_size=16 (5 cont + 11 cat/bin) | classes=3 (Low, Med, High)
    model = pred_model(input_size=(22,), classes=3)
    
    # Pour the weights into the skeleton
    # This bypasses the TypeError metadata bug entirely
    model.load_weights(str(FINAL_MODEL_PATH))
    return model

my_model = load_manual_model()

# --- PREPROCESSOR LOADING ---
with open(PRO_DATA_PATH / "preprocessor.pkl", "rb") as f:
    b = pickle.load(f)
    scalar = b["scaler"] 

def transform_input(df):
    BINARY = ['family_history_diabetes', 'family_history_heart', 'family_history_obesity']
    CONTINUOUS = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'stress_level']
    
    # Matches your categories in interface.py
    CATEGORICAL = {
        'gender':   ['Male', 'Female', 'Other'],
        'smoking':  ['Never', 'Former', 'Current'],
        'drinking': ['Never', 'Occasional', 'Regular', 'Heavy'],
        'exercise': ['Sedentary', 'Light', 'Moderate', 'Intense'],
    }

    # Scaling
    cont_vals = pd.DataFrame(
        scalar.transform(df[CONTINUOUS]),
        columns=CONTINUOUS, index=df.index
    )

    # One-Hot Encoding
    cat_dfs = []
    for col, categories in CATEGORICAL.items():
        for cat in categories:
            cat_dfs.append(pd.Series(
                (df[col] == cat).astype(int),
                name=f"{col}_{cat}", index=df.index
            ))
    cat_vals = pd.concat(cat_dfs, axis=1)
    bin_vals = df[BINARY].reset_index(drop=True)

    # Final combined input
    return pd.concat([cont_vals, cat_vals, bin_vals], axis=1)

def make_predictions(input_df):
    # Transform the raw data
    transformed = transform_input(input_df)
    
    # Get predictions from the manual model
    predictions = my_model.predict(transformed)
    
    level = ["LOW", "MEDIUM", "HIGH"] 

    # Extract indices for the 3 tasks
    diabetes_idx = np.argmax(predictions[0])
    heart_idx = np.argmax(predictions[1])
    obesity_idx = np.argmax(predictions[2])

    return {
        "diabetes": level[diabetes_idx],
        "heart": level[heart_idx],
        "obesity": level[obesity_idx]
    }