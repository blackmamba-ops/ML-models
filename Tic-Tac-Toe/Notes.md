# **Tic-Tac-Toe**


This code implements a simple reinforcement learning (RL) environment using the Tic-Tac-Toe game and the Deep Q-Network (DQN) algorithm. Here's a breakdown of what's happening in the code:

**Import Libraries:**

numpy: Numerical library for array operations.
tensorflow and keras: Libraries for building and training neural networks.
deque: Double-ended queue for storing experience replay memory.
random: Python's built-in random module.
matplotlib.pyplot: Library for data visualization.
pygame: A cross-platform set of Python modules designed for writing video games.

**Constants:**

Define constants for the Tic-Tac-Toe game and the DQN:

BOARD_SIZE: Size of the game board.
NUM_ACTIONS: Total number of possible actions (board positions).
BATCH_SIZE: Number of samples in each training batch.
MEMORY_CAPACITY: Capacity of the experience replay memory.
EPSILON_START, EPSILON_END, EPSILON_DECAY: Epsilon-greedy exploration parameters.
GAMMA: Discount factor for future rewards in Q-learning.
LEARNING_RATE: Learning rate for the optimizer.
EMPTY, PLAYER_X, PLAYER_O: Constants representing empty spaces, X player, and O player.

**TicTacToeEnvironment Class:**

Defines the Tic-Tac-Toe environment with methods for game initialization, resetting, stepping through the game, and checking for a winner or draw.

**DQNAgent Class:**

Defines the DQN agent with methods for building the DQN model, selecting actions, storing experiences, training the model, and updating the target model.

**Initialize Environment and Agent:**

Creates an instance of the TicTacToeEnvironment.
Creates an instance of the DQNAgent.

**Initialize Pygame:**

Initializes the Pygame library for rendering the Tic-Tac-Toe game.

**Gameplay Functions:**

get_random_move: Returns a random available move on the game board.
is_winner: Checks if a player has won the game.
draw: Draws the Tic-Tac-Toe game board on the screen.
handle_input: Handles player input using mouse clicks.

**Main Loop for Gameplay (main):**

Runs the main gameplay loop, alternating between player X and agent's turns.
Renders the game board using Pygame.
Checks for a winner or a draw.
Trains the agent using the DQN algorithm.

**Run the Main Loop (if name == "main"):**

Calls the main() function to start the gameplay loop.
In summary, this code combines the Tic-Tac-Toe game with a reinforcement learning agent that uses the DQN algorithm to learn optimal actions in the game. The DQN agent is trained through interactions with the environment and aims to achieve better performance over time. The Pygame library is used to visualize the game board and player interactions.


**RUN THE TIC TAC TOE FILE**

