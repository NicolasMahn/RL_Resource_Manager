import numpy as np
from tensorflow import keras
from tqdm import tqdm

from resources import util
from .prioritizing_replay_buffer import PrioritizedReplayBuffer
from .create_model import create_dqn_model

# Random number generators
rnd = np.random.random
randint = np.random.randint

verbose = 0


def prioritized_ddqn(env, episodes, gamma, epsilon, alpha, epsilon_decay, min_epsilon, rb_alpha, rb_beta, rb_beta_end,
                     batch_size, update_target_network, get_pretrained_dqn=False, progress_bar=True):
    fitness_curve = np.zeros(episodes)
    beta_increment_per_sampling = (rb_beta_end - rb_beta) / episodes

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
    replay_buffer = PrioritizedReplayBuffer(10000, rb_alpha)

    # Main training loop
    for episode in range(episodes):
        return_ = 0
        state = env.get_start_state(episode)

        # If not final state
        while not env.done(state):

            # Get Q-values for all actions from DQN
            # print("\nPredict Q values for choosing an action:")
            q_values = dqn_model.predict(np.array([state]), verbose=verbose)[0]

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

            # In Double DQN the q_values are updated with the target and the actual model
            # print("Predict Q values of next state:")
            next_q_values_from_model = dqn_model.predict(np.array([next_state]), verbose=verbose)[0]
            # Get the Q-values for the next state from the target model
            # print("Predict target Q values of next state:")
            next_q_values = target_dqn_model.predict(np.array([next_state]), verbose=verbose)[0]

            # Calculate the updated Q-value for the taken action
            q_value = q_values[action_index]
            next_q_value = next_q_values[np.argmax(next_q_values_from_model)]  # the best next q value
            td_error = (reward + gamma * next_q_value) - q_value
            q_value = td_error
            q_values[action_index] = q_value

            # Store experience to the replay buffer
            replay_buffer.add(state, q_values, action, reward, next_state,
                              td_error)

            # Start training when there are enough experiences in the buffer
            if len(replay_buffer) > batch_size:
                dqn_input, dqn_target, actions, rewards, next_states, idxs = \
                    replay_buffer.sample(batch_size, rb_beta)

                # print("Fit DQN:")
                dqn_model.fit(np.array(dqn_input), np.array(dqn_target), verbose=verbose, use_multiprocessing=True,
                              batch_size=batch_size)

                # Predict Q-values for the entire batch
                # print("Predict Q values to updated td_errors:")
                updated_q_values_batch = dqn_model.predict(np.array(dqn_input), verbose=verbose)
                # print("Predict Q values of next state to updated td_errors:")
                updated_next_q_values_from_model_batch = dqn_model.predict(np.array(next_states), verbose=verbose)
                # print("Predict target Q values of next state to updated td_errors:")
                updated_next_q_values_batch = target_dqn_model.predict(np.array(next_states), verbose=verbose)

                for i in range(len(idxs)):
                    # Select the action's Q-value for the current state
                    updated_q_value = updated_q_values_batch[i][env.action_to_int(actions[i])]

                    # Select the best next action's Q-value from the target network
                    updated_next_q_value = updated_next_q_values_batch[i][
                        np.argmax(updated_next_q_values_from_model_batch[i])]

                    updated_td_error = (rewards[i] + gamma * updated_next_q_value) - updated_q_value

                    replay_buffer.update(idxs[i], updated_td_error)

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

        # Beta encay
        rb_beta = min(rb_beta_end, rb_beta + beta_increment_per_sampling)

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
