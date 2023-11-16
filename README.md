# Reinforcement Learning Resource Management
## Abstract
This software, "Reinforcement Learning Resource Management", is an implementation focused on integrating Q-Learning, a Reinforcement Learning (RL) method, into Job Shop Scheduling (JSS). RL is a subfield of artificial intelligence, characterized by learning from interaction with an environment, using rewards or penalties. This project particularly employs Q-Learning, a model-free algorithm, allowing the agent to learn optimal policies without prior knowledge of the environment's dynamics. The core application of this technique is in JSS, illustrating its utility in scheduling tasks based on due dates and other critical parameters. The project also delves into Deep Q-Learning, extending the application's scope and efficacy in complex scheduling environments.

## Software Structure
The software comprises several key components, outlined as follows:

### Main Execution
*main.py:* Serves as the entry point of the program, functioning as a user interface for configuring settings. It lacks a graphical UI, but allows for modifications directly in the main method. Depending on these user-set variables, the main method dynamically selects and executes the appropriate algorithms.
### Packages
1. *algorithms:* Contains the DQN algorithms and the Replay Buffer. This package also defines the neural network architecture within the DQN algorithm file.
2. *environments:* Includes various RL environments, all derived from a generic_environment which establishes basic methods.
3. *resources:* Consists of helper scripts and utilities aiding the main functionality.
4. *evaluation_monitoring:* Contains scripts for evaluating and monitoring the algorithm's performance.
### Additional Folders
*models:* Generated post-execution, this folder stores the learned DQN models.
*test_sets:* Used for storing data pertinent to the evaluation of the DQN model.
## Implementation Details
This software is solely an implementation of the DQN code. It is structured to facilitate easy customization and execution of RL strategies within the realm of Job Shop Scheduling. The system's architecture is designed to be modular, allowing for flexible adaptation and scaling according to specific scheduling scenarios and requirements.