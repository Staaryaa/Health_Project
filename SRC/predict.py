import numpy as np
import pandas as pd
import keras
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from model import pred_model 

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH / "MODEL"
FINAL_MODEL_PATH = MODEL_PATH / "health_model.keras"

def load_manual_model():
    model = pred_model(input_size=(22,), classes=3)
    model.load_weights(str(FINAL_MODEL_PATH))
    return model

my_model = load_manual_model()

with open(PRO_DATA_PATH / "preprocessor.pkl", "rb") as f:
    b = pickle.load(f)
    scalar = b["scaler"] 

def transform_input(df):
    BINARY = ['family_history_diabetes', 'family_history_heart', 'family_history_obesity']
    CONTINUOUS = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'stress_level']
    

    CATEGORICAL = {
        'gender':   ['Male', 'Female', 'Other'],
        'smoking':  ['Never', 'Former', 'Current'],
        'drinking': ['Never', 'Occasional', 'Regular', 'Heavy'],
        'exercise': ['Sedentary', 'Light', 'Moderate', 'Intense'],
    }


    cont_vals = pd.DataFrame(
        scalar.transform(df[CONTINUOUS]),
        columns=CONTINUOUS, index=df.index
    )

    cat_dfs = []
    for col, categories in CATEGORICAL.items():
        for cat in categories:
            cat_dfs.append(pd.Series(
                (df[col] == cat).astype(int),
                name=f"{col}_{cat}", index=df.index
            ))
    cat_vals = pd.concat(cat_dfs, axis=1)
    bin_vals = df[BINARY].reset_index(drop=True)


    return pd.concat([cont_vals, cat_vals, bin_vals], axis=1)

def make_predictions(input_df):

    transformed = transform_input(input_df)
    

    predictions = my_model.predict(transformed)
    
    level = ["LOW", "MEDIUM", "HIGH"] 


    diabetes_idx = np.argmax(predictions[0])
    heart_idx = np.argmax(predictions[1])
    obesity_idx = np.argmax(predictions[2])

    return {
        "diabetes": level[diabetes_idx],
        "heart": level[heart_idx],
        "obesity": level[obesity_idx]

    }
