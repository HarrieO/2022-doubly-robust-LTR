# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf

def init_model(model_params):
  layers = [tf.keras.layers.Dense(x, activation='sigmoid', dtype=tf.float64)
            for x in model_params['hidden units']]
  if model_params.get('final activation', False):
    layers.append(tf.keras.layers.Dense(1, activation=tf.sigmoid, dtype=tf.float64))
  else:   
    layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float64))
  nn_model = tf.keras.Sequential(layers)
  return nn_model
