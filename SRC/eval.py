import numpy as np
import pandas as pd
import keras
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH/ "MODEL"
FINAL_MODEL_PATH = MODEL_PATH/"health_model.keras"
PLOT_PATH = ROOT_PATH/"PLOTS"

my_model = keras.models.load_model(FINAL_MODEL_PATH)

with open( PRO_DATA_PATH/"test_data.pkl" , "rb") as f:
    testing_data = pickle.load(f)

x_test = testing_data["X"]
Y_test = testing_data["y"]
y_test = [Y_test["diabetes_risk"].values , Y_test["heart_risk"].values , Y_test["obesity_risk"].values]

info = my_model.evaluate(x_test , y_test , verbose=2)

print(f"TOTAL LOSS : {info[0]}")
print(f"DIABETES LOSS : {info[1]}")
print(f"HEART DISEASE LOSS : {info[2]}")
print(f"OBSEITY LOSS : {info[3]}")
print(f"DIABETES ACCURACY : {info[4]}")
print(f"HEART DISEASE ACCURACY : {info[5]}")
print(f"OBSEITY ACCURACY : {info[6]}")

y_pred = my_model.predict(x_test)

y_pred_db = np.argmax(y_pred[0],axis=1)
y_pred_hd = np.argmax(y_pred[1],axis=1)
y_pred_ob = np.argmax(y_pred[2],axis=1)

confusion_matrix_db = confusion_matrix( y_test[0] , y_pred_db )

confusion_matrix_hd = confusion_matrix( y_test[1] ,y_pred_hd)

confusion_matrix_ob = confusion_matrix( y_test[2] ,y_pred_ob)

fig,(ax1,ax2,ax3) = plt.subplots(3,1, figsize=(8,12))

labels = ["Low", "Med", "High"]

sns.heatmap(confusion_matrix_db, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=labels, yticklabels=labels)
ax1.set_title("Diabetes Risk: Actual vs Predicted")
ax1.set_ylabel("Actual")
ax1.set_xlabel("Predicted")


sns.heatmap(confusion_matrix_hd, annot=True, fmt='d', cmap='Reds', ax=ax2, xticklabels=labels, yticklabels=labels)
ax2.set_title("Heart Risk: Actual vs Predicted")
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")


sns.heatmap(confusion_matrix_ob, annot=True, fmt='d', cmap='Greens', ax=ax3, xticklabels=labels, yticklabels=labels)
ax3.set_title("Obesity Risk: Actual vs Predicted")
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

plt.tight_layout()

plt.savefig(PLOT_PATH/"confusion_matrices.png" , dpi=300)

    