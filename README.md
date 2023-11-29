# Reinforcement Learning Resource Management
## Abstract
This software, "Reinforcement Learning Resource Management", is an implementation focused on integrating Q-Learning, a Reinforcement Learning (RL) method, into Job Shop Scheduling (JSS). RL is a subfield of artificial intelligence, characterized by learning from interaction with an environment, using rewards or penalties. This project particularly employs Q-Learning, a model-free algorithm, allowing the agent to learn optimal policies without prior knowledge of the environment's dynamics. The core application of this technique is in JSS, illustrating its utility in scheduling tasks based on due dates and other critical parameters. The project also delves into Deep Q-Learning, extending the application's scope and efficacy in complex scheduling environments.

## Software Structure
The software comprises several key components, outlined as follows:

### Main Execution
**`main.py`:** Serves as the entry point of the program, functioning as a user interface for configuring settings. It lacks a graphical UI, but allows for modifications directly in the main method. Depending on these user-set variables, the main method dynamically selects and executes the appropriate algorithms.
### Packages
1. **`algorithms`:** Contains the DQN algorithms and the Replay Buffer. This package also defines the neural network architecture within the DQN algorithm file.
2. **`environments`:** Includes various RL environments, all derived from a generic_environment which establishes basic methods.
3. **`resources`:** Consists of helper scripts and utilities aiding the main functionality.
4. **`evaluation_monitoring`:** Contains scripts for evaluating and monitoring the algorithm's performance.
### Additional Folders
**models:** Generated post-execution, this folder stores the learned DQN models.
**test_sets:** Used for storing data pertinent to the evaluation of the DQN model.
## Implementation Details
This software is solely an implementation of the DQN code. It is structured to facilitate easy customization and execution of RL strategies within the realm of Job Shop Scheduling. The system's architecture is designed to be modular, allowing for flexible adaptation and scaling according to specific scheduling scenarios and requirements.
To run this Program opptimaly a GPU should be used with linux or wsl. The other requirements can be found in the `conda_backup.yaml` or `requirements.txt`.

## Detailed Explanation

### Main File

Listed here are the functions that are executed in the main file, which mainly consists of the main function which executes all the other scripts.
The script is designed to be run as a standalone program.

**GPU Configuration:**
Sets up TensorFlow to use a specific GPU and allows dynamic memory allocation, optimizing performance for machine learning tasks. Later it verifies and displays the status of GPU devices for TensorFlow computations.

**Hyperparameters Configuration:**
Environment-specific parameters for job scheduling, like the number of machines and tasks.
DQN algorithm parameters such as the number of episodes, learning rate, and exploration factors.
Miscellaneous settings for the number of executions, model saving, and test set size (used to validate the algorithm).

**Execution Logic:**
If only one execution is set, it generates test data and trains a single DQN model.
If multiple executions are specified, it starts iterating and thus trains a new DQN model each time.
For an execution an environment ([J,m=1|nowait,f|min(T)] or [J|nowait,t|min(D)]) is set up. Then the dQN model is trained, for which a fitness curve is then later displayed.
Eventually an Example is executed and visualised with the newly trained DQN model (only if one execution is set).

### Deep Q-Network (DQN) Implementation

Deep Q-Networks merge the traditional Q-Learning approach with deep neural networks to efficiently handle complex, high-dimensional state-action spaces without the need for discretization. This makes DQN an ideal choice for our scheduling tasks, where the parameters can vary widely, creating a vast and continuous space of possibilities.
The DQN is located inside the `algorithms` package. The main algorithm is located in the `dqn.py` file, while the replay buffer class is located in the `replay_buffer.py` file.

#### Key Components

**DQN Model:** Constructed using Keras, the model comprises several dense layers activated by ReLU and a softmax output layer for action selection. The model's architecture is designed to extract deep features from the input state and provide a probability distribution over possible actions.

**Replay Buffer:** This mechanism stores experiences (state, action, reward, next state) encountered by the agent. By randomly sampling these experiences for training, the model breaks correlations between sequential experiences, leading to a more stable and effective learning process.

**Epsilon-Greedy Policy:** Balances exploration and exploitation, allowing the model to explore the environment efficiently while gradually relying more on its learned strategy.

**Experience Replay:** Enhances the efficiency of learning by using each experience in multiple updates, which is especially beneficial in environments with similar experiences.

#### Training Process

The training loop is the heart of the learning process, where the DQN model iteratively interacts with the environment. During each episode, the model updates its knowledge based on the feedback from the environment. Key steps include action selection based on the epsilon-greedy policy, reward observation, and experience storage.

The Q-values are updated using experiences from the replay buffer, employing the temporal-difference update rule. This process involves backpropagation and gradient descent to adjust the model's weights.

#### Stability and Efficiency

To ensure stability in the learning process, a target network, mirroring the DQN model, is employed. This network's weights are updated less frequently, providing a consistent target for Q-value updates and preventing the learning from becoming unstable due to constantly shifting targets.

#### Customization and Extensibility

The implementation is highly customizable, allowing for modifications in network architecture, hyperparameters, and training procedures. This flexibility ensures that the software can adapt to a wide range of scheduling scenarios.


### Environments

The environments are located in the `environments` package. Both existing environments are child classes of the `GenericEnvironment` (located in the `generic_environment.py` file). The two child environments are will be explained in further detail in the following paragraphs.

#### Environment Name

The environments are named after German job shop scheduling classification standards. Standards are defined in the [German job shop scheduling classification standards](https://de.wikipedia.org/wiki/Klassifikation_von_Maschinenbelegungsmodellen#Literatur).

##### Classifications

Under this standard, the job shop problem is first divided into 3 classifications:

<details>
  <h3>Test</h3>
  ###### α - Machine characteristics

  - **α1**: Machine type and arrangement
    - **°**: A single available machine
    - **IP**: Identical parallel machines
    - **UP**: Uniform parallel machines (with different production speeds)
    - **F**: Flow-Shop
    - **J**: Job-Shop
    - **O**: Open-Shop
  - **α2**: Number of machines
    - **°**: Any number
    - **m**: Exactly m machines

  ###### β - Task characteristics

  - **β1**: Number of tasks
    - **n=const**: A certain number of tasks is predefined. Often n=2.
    - **°**: Any number
  - **β2**: Interruptibility
    - **pmtn**: Interruption (eng. Preemption) is possible
    - **°**: No interruption
    - **nowait**: After completing a task, the next task must start immediately.
  - **β3**: Sequence relationship
    - **prec**: Predetermined sequences in the form of a graph
    - **tree**: Graph in the form of a tree
    - **°**: No sequence relationships
  - **β4**: Release time and lead time
    - **aj**: Different task release times
    - **nj**: Lead times are given. After completing a task, the task must wait a certain time before it can be processed further.
    - **°**: All tasks are available from the beginning, and there are no lead times
  - **β5**: Processing time
    - **t** refers to the duration of the processing time of the entire task or the individual tasks
    - **°**: Any processing times
  - **β6**: Sequence-dependent setup times
    - **τ**: Sequence-dependent setup time from task j to task k on machine i
    - **τb**: The tasks can be grouped into families
    - **°**: No sequence-dependent setup times
  - **β7**: Resource constraints
    - **res λσρ**
      - **λ**: Number of resources
      - **σ**: Availability of resources
      - **ρ**: Demand for resources
    - **°**: No resource constraints
  - **β8**: Completion deadlines
    - **f**: Strict deadlines are given for each task
    - **°**: No deadlines given
  - **β9**: Number of operations
    - **g**: Each task consists of exactly/at most n operations
    - **°**: Any number of operations
  - **β10**: Storage constraints
    - **κ**: Indicates the available intermediate storage for the i-th machine
    - **°**: Each machine has a storage with infinite capacity

  ###### γ - Objective

  - **D**: Minimization of throughput time
  - **Z**: Minimization of cycle time / total processing time
  - **T**: Minimization of deadline deviation
  - **V**: Minimization of tardiness
  - **L**: Minimization of idle time
</details>

#### [J,m=1|nowait,f|min(T)] Environment

The `Jm_f_T_JSSProblem` class is adept at handling situations where tasks must be completed sequentially, making it uniquely suited for problems where task dependencies and order play a significant role.

##### State Representation

In the `Jm_f_T_JSSProblem` environment, the state is represented as a list of tasks, where each element indicates a task's specific characteristics, such as its duration or priority. Unlike the `J_t_D_JSSProblem` environment, the order of tasks in this list directly impacts the agent's decision-making process, emphasizing the importance of sequence in task execution.

##### Task Scheduling Challenge

The primary challenge in this environment is to determine the most efficient order to execute tasks, considering factors like deadlines, task durations, and potential penalties for late completion. The goal is to optimize productivity by minimizing the total time taken or by maximizing the number of tasks completed within a given timeframe.

##### Actions

Actions in the `Jm_f_T_JSSProblem` environment typically involve selecting a task to execute next. Each action directly influences the state by modifying the order or the set of remaining tasks. The environment may also include actions that represent different strategies for handling task dependencies or varying levels of urgency.

##### Reward Mechanism

The reward function in the `Jm_f_T_JSSProblem` environment is designed to encourage the timely and efficient completion of tasks. Rewards are typically assigned based on factors such as meeting deadlines, optimizing task order, and efficiently utilizing time. Penalties may be incurred for late task completion or inefficient scheduling, providing a balance of positive and negative incentives to guide the learning process.

##### Application and Significance

The DQN Agent has been shown to converge (learn) when the `Jm_f_T_JSSProblem` environment was used. But since the agent, used in this environment, could be replaced with a simple sorting algorithm, its significance is limited, and it only proves that the dqn algorithm is functional.



#### [J|nowait,t|min(D)] Environment

The `J_t_D_JSSProblem` class in our software extends the concept of task scheduling in a complex and dynamic environment. This class is specifically designed to simulate a scenario where tasks, associated with specific durations, need to be allocated to multiple machines with the goal of minimizing overall execution time. This setup presents a practical instance of the classic job-shop scheduling problem, a key challenge in Operations Research.

##### State Representation

A state in the `J_t_D_JSSProblem` environment is a multidimensional array consisting of:
- A list of tasks, where each number indicates the duration of a task.
- Multiple lists representing each machine, with non-zero entries indicating assigned tasks.
- A list indicating the current time step for each machine.

For example, `[4, 3, 2, 1, 0, 0, 0, 0, 0, 0]` represents tasks to be assigned, and `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` represents an unoccupied machine.

##### Actions

Actions in this environment are defined as a list of two integers, representing the assignment of a task to a machine. A special action `[-1, -1]` is used to advance the time step, reflecting the passage of time without any new task assignment.

##### Dynamics and Transitions

The environment's transition dynamics are based on the chosen actions. For instance, assigning a task to a machine updates the machine's status and the list of pending tasks. Advancing the time step progresses the state of all machines by one time unit.

##### Initialization

The `J_t_D_JSSProblem` environment is initialized with parameters like the maximum number of machines and tasks, and task depth. These parameters define the complexity and scale of the scheduling problem. Additional settings, like `high_numb_of_tasks_preference`, allow for fine-tuning the environment to simulate various scenarios.

##### Reward Mechanism

The reward function is designed to incentivize efficient scheduling. It provides positive reinforcement for each step and penalizes uneven task distribution across machines. The goal is to encourage the agent to find a balanced and optimal task assignment.

##### Results and Implications

The results of this environment have sofar been lack luster. This environment should be seen as a work in progress
