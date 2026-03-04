import numpy as np
import pandas as pd
import keras
from keras import Model, layers
from keras.callbacks import EarlyStopping , ModelCheckpoint
from pathlib import Path
import pickle
from model import pred_model

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH/ "MODEL"
HIST_DATA_PATH = ROOT_PATH / "DATA" / "processed" / "Model_History.pkl"

with open(PRO_DATA_PATH/"train_data.pkl" , "rb") as f:
    trainning_data = pickle.load(f)

with open(PRO_DATA_PATH/"val_data.pkl" , "rb") as f:
    validation_data = pickle.load(f)

x_train = trainning_data["X"]
Y_train = trainning_data["y"]

y_train = [Y_train["diabetes_risk"].values , Y_train["heart_risk"].values , Y_train["obesity_risk"].values]

x_val = validation_data["X"]
Y_val = validation_data["y"]

y_val = [Y_val["diabetes_risk"].values , Y_val["heart_risk"].values , Y_val["obesity_risk"].values]

my_model = pred_model(input_size=(22,) , classes=3) 

early_stopping_cb = EarlyStopping(monitor="val_loss" , 
                                  mode="min" , 
                                  patience=5 , 
                                  restore_best_weights=True)

model_checkpoint_cb = ModelCheckpoint(filepath=MODEL_PATH/"health_model.keras" ,
                                      monitor="val_loss" , 
                                      mode="min" , 
                                      save_best_only=True , 
                                      save_weights_only=False )

my_model.compile(optimizer="adam" , metrics={
    "diabetes": "accuracy",
    "heart_risk": "accuracy",
    "obesity": "accuracy"
} , loss=["sparse_categorical_crossentropy" , "sparse_categorical_crossentropy",  "sparse_categorical_crossentropy"])

model_info  = my_model.fit(x=x_train ,
                            y=y_train ,
                            validation_data=(x_val , y_val),
                            epochs = 100,
                            batch_size=32,
                            callbacks=[early_stopping_cb , model_checkpoint_cb]
                           )

with open(HIST_DATA_PATH ,"wb") as f :
    pickle.dump(model_info.history ,f)



