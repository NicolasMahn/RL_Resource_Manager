import numpy as np
from resources import util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
import sys
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


def create_dqn_model(dimensions, num_actions, learning_rate=0.00025):
    state_input = tf.keras.layers.Input(shape=(dimensions,))

    # Convolutions on the frames on the screen
    layer1 = layers.Dense(32, activation="relu")(state_input)
    layer2 = layers.Dense(64, activation="relu")(layer1)
    layer3 = layers.Dense(64, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    model = keras.Model(inputs=state_input, outputs=action)

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['mae'])

    return model, optimizer, loss_function


def q_learning(env, episodes=500, gamma=0.9, epsilon=0.4, alpha=0.1, epsilon_decay=0.9, updates=False):
    fitness_curve = list()

    # Create a progress bar
    # custom_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
    progress_bar = tqdm(total=episodes, unit='episode')

    # Q-Table
    dqn_model, optimizer, loss_function = create_dqn_model(len(env.dimensions), len(env.actions), learning_rate=alpha)

    # the main training loop
    for episode in range(episodes + 1):
        stdout = redirect_stdout()


        # initial state
        state = env.get_start_state()

        return_ = 0

        iter = 0
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
            q_values[0][impossible_action_indices] = -1e6  # a large negative value

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
            updated_q_values = reward + gamma * np.max(next_q_values)
            done_mask = 0.0 if env.done(next_state) else 1.0
            updated_q_values = updated_q_values * done_mask - (1 - done_mask)

            # Create a mask for the action taken
            action_mask = tf.one_hot(env.action_to_int(action), len(env.actions))

            with tf.GradientTape() as tape:
                # Get Q-values from the model
                q_values = dqn_model(np.array([env.state_for_dqn(state)]))

                # Apply the mask to get the Q-value of the action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, action_mask), axis=1)

                # Calculate loss
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, dqn_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, dqn_model.trainable_variables))

            iter += 1
            if iter > 50 and updates:  # change this for debugging
                restore_stdout(stdout)
                print("\n-------------------------------------")
                print(f"episode {episode}, iteration {iter}")
                print(f"state: {list(state)}")
                print(f"action: {list(action)}")
                print(f"next state: {next_state}")
                print("-------------------------------------")
                stdout = redirect_stdout()

            # Update state
            state = list(next_state)


        fitness_curve.append(return_)

        epsilon *= epsilon_decay

        # Update the progress bar
        restore_stdout(stdout)
        progress_bar.update(1)


    # Close the progress bar
    progress_bar.close()

    return dqn_model, fitness_curve
