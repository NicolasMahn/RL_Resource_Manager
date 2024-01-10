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


def redirect_stdout():
    # Redirect stdout to /dev/null or null to suppress output
    stdout_orig = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return stdout_orig


def restore_stdout(stdout_orig):
    # Restore the original stdout after redirection
    sys.stdout.close()
    sys.stdout = stdout_orig


def get_pi_from_q(env, dqn_model, initial_state, less_comments=False):
    # If toggled the standard output is restricted to suppress unwanted output
    if less_comments:
        stdout = redirect_stdout()

    optimal_policy = []

    # Initial state
    state = initial_state

    while not env.done(state):
        # Get Q-values for all actions from DQN model
        q_values = dqn_model.predict(np.array([env.to_tensor_state(state)]))

        # Determine possible and impossible actions
        possible_actions, impossible_actions = env.get_possible_actions(state)
        if len(possible_actions) == 0:
            print("PROBLEM: No action possible")
            break
        possible_action_indices = [env.action_to_int(action) for action in possible_actions]
        impossible_action_indices = [env.action_to_int(action) for action in impossible_actions]

        # Mask Q-values of impossible actions
        q_values[0][impossible_action_indices] = -np.inf  # Replace with a large negative value

        # Greedy action selection
        action = env.actions[util.argmax(q_values[0])]

        optimal_policy.append((list(state), list(action)))

        # Get next state
        state = env.get_next_state(state, action)

    # Add the final state
    optimal_policy.append((state, None))

    # If toggled the standard output is restored
    if less_comments:
        restore_stdout(stdout)

    return optimal_policy


def create_dqn_model(dimensions, numb_of_actions):
    # Define input and dense layers of the DQN model
    state_input = keras.layers.Input(shape=tuple(dimensions))
    layer1 = keras.layers.Dense(64, activation="relu")(state_input)
    layer2 = keras.layers.Dense(128, activation="relu")(layer1)
    flat_layer = keras.layers.Flatten()(layer2)  # Flatten the layers
    layer3 = keras.layers.Dense(128, activation="relu")(flat_layer)
    layer4 = keras.layers.Dense(64, activation="relu")(layer3)
    action = keras.layers.Dense(numb_of_actions, activation="softmax")(layer4)

    # Construct and compile the model
    model = keras.Model(inputs=state_input, outputs=action)
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.Huber()

    return model, optimizer, loss_function


def q_learning(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon, batch_size, update_target_network,
               get_pretrained_dqn=False, progress_bar=True, get_histories=False):
    fitness_curve = list()
    histories = list()

    # Create a progress bar for training
    if progress_bar:
        progress_bar = tqdm(total=episodes, unit='episode')

    # Initialize DQN and target models
    dqn_model, optimizer, loss_function = create_dqn_model(env.dimensions, len(env.actions))
    target_dqn_model = keras.models.clone_model(dqn_model)
    target_dqn_model.set_weights(dqn_model.get_weights())

    # Initialize pretrained model
    pretrained_dqn_model = keras.models.clone_model(dqn_model)
    pretrained_dqn_model.set_weights(dqn_model.get_weights())

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(10000)

    # Main training loop
    for episode in range(episodes):
        state = env.get_start_state(episode)
        return_ = 0
        history = list()

        # If not final state
        while not env.done(state):

            # Get Q-values for all actions from DQN
            actual_q_values = dqn_model(
                np.array([env.to_tensor_state(state)]))
            q_values = actual_q_values.numpy()[0]

            # Action selection and masking
            possible_actions, impossible_actions = \
                env.get_possible_actions(state)
            if len(possible_actions) == 0:
                print("PROBLEM: No action possible")
                break
            possible_action_indices = \
                [env.action_to_int(action) for action in possible_actions]
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

            # Store experience to the replay buffer
            replay_buffer.push(env.to_tensor_state(state), action_index, reward, env.to_tensor_state(next_state))

            # Start training when there are enough experiences in the buffer
            if len(replay_buffer) > batch_size:
                replay_state, replay_action, \
                    replay_reward, replay_next_state = \
                    replay_buffer.sample(batch_size)

                # Update Q-values using the experiences from the replay buffer
                for i in range(batch_size):
                    replay_next_q_values = \
                        target_dqn_model(np.array([replay_next_state[i]])
                                         ).numpy()[0]
                    replay_updated_q_values = \
                        dqn_model(np.array([replay_state[i]])).numpy()[0]
                    replay_updated_q_values[replay_action[i]] = \
                        replay_updated_q_values[replay_action[i]] + alpha * (
                                (replay_reward[i] + gamma * np.max(replay_next_q_values)) -
                                replay_updated_q_values[replay_action[i]])
                    action_mask = tf.one_hot(replay_action[i], len(env.actions))

                    with tf.GradientTape() as tape:
                        qv = dqn_model(np.array([replay_state[i]]))
                        q_action = tf.reduce_sum(tf.multiply(qv, action_mask), axis=1)
                        loss = loss_function(replay_updated_q_values, q_action)

                    grads = tape.gradient(loss, dqn_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, dqn_model.trainable_variables))

            history.append({
                "state": state,
                "action": action
            })

            # Update state
            state = list(next_state)

        # Target network update
        if episode % update_target_network == 0:
            target_dqn_model.set_weights(dqn_model.get_weights())

        fitness_curve.append(return_)
        histories.append(history)

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update the progress bar
        if progress_bar:
            progress_bar.update(1)

    # Close the progress bar
    if progress_bar:
        progress_bar.close()

    # Return models and fitness curve
    if get_pretrained_dqn and get_histories:
        return dqn_model, fitness_curve, pretrained_dqn_model, histories
    elif get_pretrained_dqn:
        return dqn_model, fitness_curve, pretrained_dqn_model
    elif get_histories:
        return dqn_model, fitness_curve, histories
    else:
        return dqn_model, fitness_curve
