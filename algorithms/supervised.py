import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import sys
import pandas as pd

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

def create_dataset(env, epochs):
    dataset = []

    for epoch in range(epochs):
        state = env.get_start_state(epoch)

        while not env.done(state):

            # Action selection and masking
            possible_actions, impossible_actions = env.get_possible_actions(state)
            if len(possible_actions) == 0:
                break

            chosen_action_index = randint(0, (len(possible_actions)))
            action = possible_actions[chosen_action_index]
            action_index = env.action_to_int(action)

            # Take action, observe next state and correctness
            next_state = env.get_next_state(state, action)
            correct = env.check_if_step_correct(state, action, next_state)

            # Save data to dataset
            dataset.append((padd_state(env, state), action_index, correct))

            # Do not run already broken examples
            if not correct:
                break

            # Update state
            state = list(next_state)

    # Convert dataset to DataFrame for easier manipulation
    df = pd.DataFrame(dataset, columns=['state', 'action', 'correct'])
    return df


def preprocess_data(env, df):
    # for i, state in enumerate(df['state']):
    #    print(f"Index {i}, Shape: {np.array(state).shape}")
    X = np.stack(df['state'].tolist())
    y = keras.utils.to_categorical(df['action'], num_classes=len(env.actions))
    return X, y


def create_nn_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_actions, activation='softmax')  # num_actions should be 81 in your case
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def supervised_learning(env, epochs, batch_size, get_pretrained_dnn=False):
    # Generate and preprocess the dataset
    dataset_df = create_dataset(env, epochs)
    X, y = preprocess_data(env, dataset_df)
    print(y.shape)
    print(y)

    # Create and train the neural network model
    model = create_nn_model(X.shape[1:], len(env.actions))

    pretrained_nn_model = keras.models.clone_model(model)
    pretrained_nn_model.set_weights(model.get_weights())

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size)


    if get_pretrained_dnn:
        return model, history, pretrained_nn_model
    else:
        return model, history


