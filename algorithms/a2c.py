import numpy as np
from tqdm import tqdm

from .create_model import create_critic_model, create_actor_model

# Random number generators
rnd = np.random.random
randint = np.random.randint


def a2c(env, episodes, gamma, alpha, progress_bar=True):
    fitness_curve = np.zeros(episodes)

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
            action_probabilities = actor_model.predict(np.array([state]), verbose=0)[0]

            # Choose an action based on the modified probabilities
            action_index = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            action = env.actions[action_index]

            # Take action, observe reward and next state
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            return_ += reward

            critic_value = critic_model.predict(np.array([state]), verbose=0)[0]
            next_critic_value = critic_model.predict(np.array([next_state]), verbose=0)[0]

            target = reward + (gamma * next_critic_value * (1 - int(env.done(state))))
            delta = target - critic_value

            actions = np.zeros([1, len(env.actions)])
            actions[np.arange(1), action_index] = 1

            # Update critic
            critic_model.fit(np.array([state]), target, verbose=0)

            # Update actor
            actor_model.fit(np.array([state]), actions, sample_weight=delta.flatten(), verbose=0)

            # Update state
            state = next_state.copy()

            if progress_bar:
                progress_bar.refresh()

        fitness_curve[episode] = return_

        # Update the progress bar
        if progress_bar:
            progress_bar.update(1)

    # Close the progress bar
    if progress_bar:
        progress_bar.close()

    # Return models and fitness curve
    return actor_model, critic_model, fitness_curve

