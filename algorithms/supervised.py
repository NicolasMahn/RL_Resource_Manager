import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import sys
import pandas as pd

import resources.data_generation as dg
from resources import util
from .replay_buffer import ReplayBuffer

# Random number generators
rnd = np.random.random
randint = np.random.randint


def redirect_stdout():
    # Redirect stdout to /dev/null or null to suppress output
    stdout_orig = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return stdout_orig

def restore_stdout(stdout_orig):
    # Restore the original stdout after redirection
    sys.stdout.close()
    sys.stdout = stdout_orig


def padd_state(env, state):
    state_padded = list()
    for s in state:
        s_padded = np.pad(s, (0, env.max_numb_of_tasks - len(s)), constant_values=-1)
        state_padded.append(s_padded)
    return state_padded


def create_nn_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(num_actions, activation='softmax')  # num_actions should be 81 in your case
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def supervised_learning(env, episodes, epochs, batch_size, get_pretrained_dnn=False):
    # Generate and preprocess the dataset
    dataset = list(dg.create_correct_histories(env, episodes, result_as_unsorted_state_action_pairs=True))
    X, y = util.preprocess_data(env, dataset)

    # Create and train the neural network model
    model = create_nn_model(X.shape[1:], len(env.actions))

    pretrained_nn_model = keras.models.clone_model(model)
    pretrained_nn_model.set_weights(model.get_weights())

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size)


    if get_pretrained_dnn:
        return model, history, pretrained_nn_model
    else:
        return model, history


