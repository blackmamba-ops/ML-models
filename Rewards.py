import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Constants
BOARD_SIZE = 3
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.95
LEARNING_RATE = 0.001

# Define the DQN model
def build_dqn_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
    return model

# Create the DQN agent
class DQNAgent:
    def __init__(self):
        self.model = build_dqn_model(input_shape=(NUM_ACTIONS,), num_actions=NUM_ACTIONS)
        self.target_model = build_dqn_model(input_shape=(NUM_ACTIONS,), num_actions=NUM_ACTIONS)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(NUM_ACTIONS)
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        targets = self.model.predict(np.array(states))
        next_q_values = self.target_model.predict(np.array(next_states))
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])
        self.model.fit(np.array(states), np.array(targets), verbose=0)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Define the Tic-Tac-Toe environment
class TicTacToeEnvironment:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, BOARD_SIZE)
        if self.board[row, col] != 0:
            return self.board.flatten(), 0, False
        self.board[row, col] = self.current_player
        reward = self.check_winner(self.current_player)
        done = reward != 0 or np.all(self.board != 0)
        self.current_player = -self.current_player
        return self.board.flatten(), reward, done

    def check_winner(self, player):
        for i in range(BOARD_SIZE):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return 1 if player == 1 else -1
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return 1 if player == 1 else -1
        return 0

# Initialize the environment and agent
env = TicTacToeEnvironment()
agent = DQNAgent()

# Training loop
rewards_history = []

for episode in range(50):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state

    # Decay epsilon to encourage exploitation over time
    agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

    rewards_history.append(total_reward)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Plot the rewards history
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.grid()
plt.show()