import numpy as np
import util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
import sys
import math
from tensorflow import keras
from tensorflow.keras import layers

random = np.random.random
randint = np.random.randint



def redirect_stdout():
    # Redirect stdout to /dev/null or nul
    stdout_orig = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return stdout_orig


def restore_stdout(stdout_orig):
    # Restore the original stdout
    sys.stdout.close()
    sys.stdout = stdout_orig

def get_pi_from_q(env, dqn_model):
    stdout = redirect_stdout()
    optimal_policy = []

    # initial state
    state = env.get_start_state()

    while not env.done(state):
        # Get Q-values for all actions from DQN
        q_values = dqn_model.predict(np.array([env.state_for_dqn(state), ]))

        # choose a possible action
        possible_actions, impossible_actions = env.get_possible_actions(state)
        if len(possible_actions) == 0:
            break
        possible_action_indices = [env.action_to_int(action) for action in possible_actions]
        impossible_action_indices = [env.action_to_int(action) for action in impossible_actions]

        # Mask Q-values of impossible actions
        q_values[0][impossible_action_indices] = -np.inf  # Replace with a large negative value

        # greedy
        action = env.actions[util.argmax(q_values[0])]

        optimal_policy.append((list(state), list(action)))

        state = env.get_next_state(state, action)

    optimal_policy.append((state, None)) # Add the final state

    restore_stdout(stdout)
    return optimal_policy


def create_dqn_model(num_states, num_actions):
    state_input = tf.keras.layers.Input(shape=(num_states,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(state_input)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    model =  keras.Model(inputs=inputs, outputs=action)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0), loss='mse',  metrics=['mae'])

    return model


def q_learning(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.9, updates=False):
    fitness_curve = list()

    # Create a progress bar
    # custom_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    progress_bar = tqdm(total=episodes, unit='episode')

    # Q-Table
    dqn_model = create_dqn_model(len(env.dimensions), len(env.actions))

    # the main training loop
    for episode in range(episodes + 1):
        stdout = redirect_stdout()


        # initial state
        state = env.get_start_state()

        return_ = 0

        # if not final state
        while not env.done(state):

            # Get Q-values for all actions from DQN
            q_values = dqn_model.predict(np.array([env.state_for_dqn(state), ]))

            # choose a possible action
            possible_actions, impossible_actions = env.get_possible_actions(state)
            if len(possible_actions) == 0:
                break
            possible_action_indices = [env.action_to_int(action) for action in possible_actions]
            impossible_action_indices = [env.action_to_int(action) for action in impossible_actions]

            # Mask Q-values of impossible actions
            q_values[0][impossible_action_indices] = -1e6  # Replace with a large negative value

            # Choose action based on epsilon-greedy policy
            if random() < epsilon:
                # choose random action
                action = possible_actions[randint(0, len(possible_actions))]
            else:
                # greedy
                action = env.actions[util.argmax(q_values[0])]

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            # Get Q-values for next state
            next_q_values = dqn_model.predict(np.array([env.state_for_dqn(next_state)]))

            # Calculate target Q-value for the taken action
            td_target_estimate = reward + gamma * np.max(next_q_values)
            td_error = td_target_estimate - q_values[0][env.action_to_int(action)]
            target_q_value = q_values[0][env.action_to_int(action)] + alpha * td_error
            if math.isnan(target_q_value):
                raise ValueError("NaN in target Q-value")
            targets = q_values
            targets[0][env.action_to_int(action)] = target_q_value

            # Perform one gradient descent step on (state, targets)
            dqn_model.fit(np.array([env.state_for_dqn(state)]), targets, verbose=0)

            # Update state
            state = next_state

        fitness_curve.append(return_)

        epsilon *= epsilon_decay

        # Update the progress bar
        restore_stdout(stdout)
        progress_bar.update(1)


    # Close the progress bar
    progress_bar.close()

    return dqn_model, fitness_curve
