import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

OUT_DIR = Path("DATA/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(r".\DATA\health_data_synthetic.csv")

BINARY     = ['family_history_diabetes', 'family_history_heart', 'family_history_obesity']

CONTINUOUS = [
    'age',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'stress_level'
]

CATEGORICAL = {
    'gender':   ['Male', 'Female', 'Other'],
    'smoking':  ['Never', 'Former', 'Current'],
    'drinking': ['Never', 'Occasional', 'Regular', 'Heavy'],
    'exercise': ['Sedentary', 'Light', 'Moderate', 'Intense'],
}

TARGETS = ['diabetes_risk', 'heart_risk', 'obesity_risk']

RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2}

scaler = StandardScaler()
X_cont = pd.DataFrame(
    scaler.fit_transform(df[CONTINUOUS]),
    columns=CONTINUOUS, index=df.index
)

cat_dfs = []
for col, categories in CATEGORICAL.items():
    for cat in categories:
        cat_dfs.append(pd.Series(
            (df[col] == cat).astype(int),
            name=f"{col}_{cat}", index=df.index
        ))
X_cat = pd.concat(cat_dfs, axis=1)

X_bin = df[BINARY].reset_index(drop=True)
X_cont = X_cont.reset_index(drop=True)
X_cat  = X_cat.reset_index(drop=True)

x = pd.concat([X_cont, X_cat, X_bin], axis=1)

y_raw = df[TARGETS].reset_index(drop=True)
y = y_raw.map(lambda v: RISK_MAP[v])

x_train , x_val_test , y_train , y_val_test = train_test_split(x,y, test_size = 0.3 , random_state=23 , stratify=y['heart_risk'])
x_val , x_test , y_val , y_test = train_test_split(x_val_test, y_val_test , test_size = 0.5 , random_state=23 , stratify=y_val_test['heart_risk'])

datasets = {
    "train": {"X": x_train, "y": y_train},
    "val":   {"X": x_val, "y": y_val},
    "test":  {"X": x_test, "y": y_test}
}

for name, data in datasets.items():
    file_path = OUT_DIR / f"{name}_data.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

with open(OUT_DIR / "preprocessor.pkl", 'wb') as f:
    pickle.dump({'scaler': scaler, 'feature_names': list(x.columns)}, f)

