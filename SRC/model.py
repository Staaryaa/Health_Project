import numpy as np
import pandas as pd
import keras
from keras import Model, layers

def pred_model(input_size , classes):

    input_layer = layers.Input(shape=input_size)

    layer_dense = layers.Dense(256, activation="relu")(input_layer)
    layer_norm = layers.BatchNormalization()(layer_dense)
    layer_drop = layers.Dropout(0.3)(layer_norm)

    layer_dense = layers.Dense(128, activation="relu")(layer_drop)
    layer_norm = layers.BatchNormalization()(layer_dense)
    layer_drop = layers.Dropout(0.3)(layer_norm)

    layer_dense = layers.Dense(64, activation="relu")(layer_drop)
    layer_norm = layers.BatchNormalization()(layer_dense)
    layer_drop = layers.Dropout(0.3)(layer_norm)

    layer_dense = layers.Dense(32, activation="relu")(layer_drop)
    layer_norm = layers.BatchNormalization()(layer_dense)
    layer_drop = layers.Dropout(0.3)(layer_norm)

    layer_db = layers.Dense(16, activation = "relu")(layer_drop)
    layer_db_output = layers.Dense(classes , activation="softmax" , name ="diabetes")(layer_db)

    layer_hd = layers.Dense(16, activation = "relu")(layer_drop)
    layer_hd_output = layers.Dense(classes , activation="softmax" , name ="heart_risk")(layer_hd)

    layer_ob = layers.Dense(16, activation = "relu")(layer_drop)
    layer_ob_output = layers.Dense(classes , activation="softmax" , name ="obesity")(layer_ob)

    model = Model(inputs = input_layer , outputs = [layer_db_output ,layer_hd_output, layer_ob_output ])

    return model


