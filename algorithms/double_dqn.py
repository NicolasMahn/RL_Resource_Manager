import numpy as np
from tensorflow import keras
from tqdm import tqdm

from resources import util
from .replay_buffer import ReplayBuffer
from .create_model import create_dqn_model

# Random number generators
rnd = np.random.random
randint = np.random.randint


def ddqn(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon, batch_size, update_target_network,
         get_pretrained_dqn=False, progress_bar=True):
    fitness_curve = np.zeros(episodes)

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

            # Get Q-values for all actions from DQN
            q_values = dqn_model.predict(np.array([state]), verbose=0)[0]

            # Epsilon-greedy policy
            if rnd() < epsilon:
                action = env.actions[randint(0, len(env.actions))]
            else:
                action = env.actions[util.argmax(q_values)]
            action_index = env.action_to_int(action)

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            next_state_for_model = np.array([next_state])
            # In Double DQN the q_values are updated with the target and the actual model
            next_q_values_from_model = dqn_model.predict(next_state_for_model, verbose=0)[0]
            # Get the Q-values for the next state from the target model
            next_q_values = target_dqn_model.predict(next_state_for_model, verbose=0)[0]

            # Calculate the updated Q-value for the taken action
            q_value = q_values[action_index]
            next_q_value = next_q_values[np.argmax(next_q_values_from_model)]  # the best next q value
            q_value = (reward + gamma * next_q_value) - q_value
            q_values[action_index] = q_value

            # Store experience to the replay buffer
            replay_buffer.push(state, q_values)

            # Start training when there are enough experiences in the buffer
            if len(replay_buffer) > batch_size:
                dqn_input, dqn_output = replay_buffer.sample(batch_size)
                dqn_model.fit(np.array(dqn_input), np.array(dqn_output), verbose=0, use_multiprocessing=True,
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
