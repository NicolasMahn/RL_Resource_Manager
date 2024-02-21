import numpy as np
import os
from tensorflow import keras
import sys


import resources.data_generation as dg
from resources import util
from .create_model import create_dqn_model


# Random number generators
rnd = np.random.random
randint = np.random.randint


def supervised_learning(env, episodes, epochs, alpha, batch_size, get_pretrained_dnn=False):
    # Generate and preprocess the dataset
    dataset = list(dg.create_correct_histories(env, episodes, result_as_unsorted_state_action_pairs=True))
    x, y = util.preprocess_data(env, dataset)

    # Create and train the neural network model
    model = create_dqn_model(x.shape[1:], len(env.actions), alpha)

    pretrained_nn_model = keras.models.clone_model(model)
    pretrained_nn_model.set_weights(model.get_weights())

    history = model.fit(x, y, epochs=epochs, batch_size=batch_size)

    if get_pretrained_dnn:
        return model, history, pretrained_nn_model
    else:
        return model, history


