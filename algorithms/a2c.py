import numpy as np
from tensorflow import keras
from tqdm import tqdm

from resources import util
from .replay_buffer import ReplayBuffer
from .create_model import create_critic_model, create_actor_model

# Random number generators
rnd = np.random.random
randint = np.random.randint


def a2c(env, episodes, gamma, alpha, progress_bar=True):
    fitness_curve = list()

    # Create a progress bar for training
    if progress_bar:
        progress_bar = tqdm(total=episodes, unit='episode')

    actor_model = create_actor_model(env.dimensions, len(env.actions), alpha)
    critic_model = create_critic_model(env.dimensions, alpha)

    # Main training loop
    for episode in range(episodes):
        return_ = 0
        state = env.get_start_state(episode)

        # If not final state
        while not env.done(state):


            # choose an action
            action_probabilities = actor_model.predict(np.array([env.to_tensor_state(state)]), verbose=0)[0]
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            action_index = env.action_to_int(action)

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            critic_value = critic_model.predict(np.array([env.to_tensor_state(state)]), verbose=0)[0]
            next_critic_value = critic_model.predict(next_state)

            target = reward + (gamma * next_critic_value * (1 - int(env.done(state))))
            delta = target - critic_value

            actions = np.zeros([1, env.actions])
            actions[np.arange(1), action_index] = 1

            if progress_bar:
                progress_bar.refresh()

            # Update critic
            critic_model.fit(np.array([env.to_tensor_state(state)]), target, verbose=0)

            if progress_bar:
                progress_bar.refresh()

            # Update actor
            actor_model.fit(np.array([env.to_tensor_state(state)]), actions, sample_weight=delta.flatten(), verbose=0)

            if progress_bar:
                progress_bar.refresh()

            # Update state
            state = list(next_state)

        fitness_curve.append(return_)

        # Update the progress bar
        if progress_bar:
            progress_bar.update(1)

    # Close the progress bar
    if progress_bar:
        progress_bar.close()

    # Return models and fitness curve
    return actor_model, critic_model, fitness_curve

