import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import sys

from resources import util
from .replay_buffer import ReplayBuffer

# Random number generators
rnd = np.random.random
randint = np.random.randint


def create_dqn_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(num_actions, activation="linear")
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def q_learning(env, episodes, epochs, gamma, epsilon, alpha, epsilon_decay, min_epsilon, batch_size,
               update_target_network, get_pretrained_dqn=False, progress_bar=True):
    fitness_curve = list()

    # Create a progress bar for training
    if progress_bar:
        progress_bar = tqdm(total=episodes, unit='episode')

    # Initialize DQN and target models
    dqn_model = create_dqn_model(env.dimensions, len(env.actions))
    target_dqn_model = keras.models.clone_model(dqn_model)
    target_dqn_model.set_weights(dqn_model.get_weights())

    # Initialize pretrained model
    pretrained_dqn_model = keras.models.clone_model(dqn_model)
    pretrained_dqn_model.set_weights(dqn_model.get_weights())

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(10000)

    # Main training loop
    for episode in range(episodes):
        return_ = 0
        state = env.get_start_state(episode)

        # If not final state
        while not env.done(state):

            # Get Q-values for all actions from DQN
            actual_q_values = dqn_model.predict(np.array([env.to_tensor_state(state)]), verbose=0)[0]
            q_values = np.array(list(actual_q_values))

            # Action selection and masking
            possible_actions, impossible_actions = \
                env.get_possible_actions(state)
            if len(possible_actions) == 0:
                print("PROBLEM: No action possible")
                break
            impossible_action_indices = \
                [env.action_to_int(action) for action in impossible_actions]

            q_values[impossible_action_indices] = -1e6  # Mask with a large negative value

            # Epsilon-greedy policy
            if rnd() < epsilon:
                action = possible_actions[randint(0, len(possible_actions))]
            else:
                action = env.actions[util.argmax(q_values)]
                if action in impossible_actions:
                    action = possible_actions[randint(0, len(possible_actions))]
            action_index = env.action_to_int(action)

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            # Get the Q-values for the next state from the target model
            next_state_for_model = np.array([env.to_tensor_state(next_state)])
            next_q_values = target_dqn_model.predict(next_state_for_model)[0]

            # Calculate the updated Q-value for the taken action
            q_values = actual_q_values
            q_value = q_values[action_index]
            q_value = q_value + alpha * ((reward + gamma * np.max(next_q_values)) - q_value)
            q_values[action_index] = q_value

            # Store experience to the replay buffer
            replay_buffer.push(env.to_tensor_state(state), q_values)

            # Start training when there are enough experiences in the buffer
            if len(replay_buffer) > batch_size:
                X, y = replay_buffer.sample(batch_size)

                if progress_bar:
                    progress_bar.refresh()

                dqn_model.fit(np.array(X), np.array(y), verbose=0, epochs=epochs, use_multiprocessing=True,
                              batch_size=batch_size)

                if progress_bar:
                    progress_bar.refresh()

            # Update state
            state = list(next_state)

        # Target network update
        if episode % update_target_network == 0:
            target_dqn_model.set_weights(dqn_model.get_weights())

        fitness_curve.append(return_)

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update the progress bar
        if progress_bar:
            progress_bar.update(1)

    # Close the progress bar
    if progress_bar:
        progress_bar.close()

    # Return models and fitness curve
    if get_pretrained_dqn:
        return dqn_model, fitness_curve, pretrained_dqn_model
    else:
        return dqn_model, fitness_curve
