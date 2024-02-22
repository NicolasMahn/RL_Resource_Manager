import numpy as np
from tensorflow import keras
from tqdm import tqdm

from resources import util
from .replay_buffer import ReplayBuffer
from .create_model import create_dqn_model

# Random number generators
rnd = np.random.random
randint = np.random.randint


def dqn(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon, batch_size, update_target_network,
        get_pretrained_dqn=False, progress_bar=True):
    fitness_curve = np.zeros(episodes)

    # Initialize dqn_input and dqn_target with the correct shapes
    dqn_input = np.zeros((batch_size,) + tuple(env.dimensions))
    dqn_target = np.zeros((batch_size, len(env.actions)))

    # Create a progress bar for training
    if progress_bar:
        progress_bar = tqdm(total=episodes, unit='episode')

    # Initialize DQN and target models
    dqn_model = create_dqn_model(env.dimensions, len(env.actions), alpha)
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

            # Action selection and masking
            possible_actions, impossible_actions = \
                env.get_possible_actions(state, index=True)

            if len(possible_actions) == 0:
                print("PROBLEM: No action possible")
                break

            # Epsilon-greedy policy
            if rnd() < epsilon:
                action_index = possible_actions[randint(0, len(possible_actions))]
            else:
                q_values = dqn_model.predict(np.array([state]), verbose=0)[0]
                if len(impossible_actions) > 0:
                    q_values[impossible_actions] = -1e6  # Mask with a large negative value
                action_index = util.argmax(q_values)
                if action_index in impossible_actions:
                    action_index = possible_actions[randint(0, len(possible_actions))]
            action = env.actions[action_index]

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            # Store experience to the replay buffer
            replay_buffer.push(state, action_index, reward, next_state)

            # Start training when there are enough experiences in the buffer
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)

                states = np.array([b_state for b_state, _, _, _ in batch])
                next_states = np.array([b_next_state for _, _, _, b_next_state in batch])

                q_values_batch = target_dqn_model.predict(states, verbose=0)
                next_q_values_batch = target_dqn_model.predict(next_states, verbose=0)

                for i, (b_state, b_action, b_reward, b_next_state) in enumerate(batch):
                    # Get the Q-values of the state, and next state from the target model
                    q_values = q_values_batch[i]
                    next_q_values = next_q_values_batch[i]

                    # Calculate the updated Q-value for the taken action
                    q_value = q_values[b_action]
                    q_value = (b_reward + gamma * np.max(next_q_values)) - q_value
                    q_values[b_action] = q_value

                    dqn_input[i] = np.array(state)
                    dqn_target[i] = q_values

                dqn_model.fit(np.array(dqn_input), np.array(dqn_target),
                              verbose=0, use_multiprocessing=True,
                              batch_size=batch_size)

            # Update state
            state = next_state.copy()

            if progress_bar:
                progress_bar.refresh()

        # Target network update
        if episode % update_target_network == 0:
            target_dqn_model.set_weights(dqn_model.get_weights())

        fitness_curve[episode] = return_

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
