import sys
import numpy as np
import matplotlib.pyplot as plt

# Constants
ROWS = 4
COLS = 4
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 1000

# Initialize Q-table
q_table = np.zeros((ROWS, COLS, len(ACTIONS)))

# Define the Frozen Lake environment with states and rewards
frozen_lake = [
    ["S", "F", "F", "F"],
    ["F", "H", "F", "H"],
    ["F", "F", "F", "H"],
    ["H", "F", "F", "G"]
]

def get_next_state(state, action):
    row, col = state
    if action == "UP" and row > 0:
        return (row - 1, col)
    elif action == "DOWN" and row < ROWS - 1:
        return (row + 1, col)
    elif action == "LEFT" and col > 0:
        return (row, col - 1)
    elif action == "RIGHT" and col < COLS - 1:
        return (row, col + 1)
    return (row, col)

def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.choice(ACTIONS) # Exploration: Choose a random action with probability EPSILON
    return ACTIONS[np.argmax(q_table[state])] # Exploitation: Choose the action with the highest Q-value for the current state(expected cumilative reward)

def update_q_table(state, action, next_state, reward):
    predict = q_table[state][ACTIONS.index(action)]
    target = reward + GAMMA * np.max(q_table[next_state])
    q_table[state][ACTIONS.index(action)] += ALPHA * (target - predict)

def visualize_path(path):
    plt.figure(figsize=(COLS, ROWS))
    for i in range(ROWS):
        for j in range(COLS):
            cell = frozen_lake[i][j]
            if cell == "H":
                plt.text(j + 0.5, ROWS - i - 0.5, "H", fontsize=12, color="red", ha="center", va="center")
            elif cell == "G":
                plt.text(j + 0.5, ROWS - i - 0.5, "G", fontsize=12, color="green", ha="center", va="center")
            elif cell == "S":
                plt.text(j + 0.5, ROWS - i - 0.5, "S", fontsize=12, color="blue", ha="center", va="center")
            else:
                plt.text(j + 0.5, ROWS - i - 0.5, "F", fontsize=12, color="black", ha="center", va="center")
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]
        plt.arrow(current_state[1], ROWS - current_state[0] - 1,
                  next_state[1] - current_state[1], current_state[0] - next_state[0],
                  head_width=0.1, head_length=0.1, color="blue")
    plt.xticks(range(COLS), [""] * COLS)
    plt.yticks(range(ROWS), [""] * ROWS)
    plt.grid(visible=True)
    plt.show()

def train():
    rewards = []
    paths = []  # List to store visited states for each episode
    for episode in range(EPISODES):
        state = (0, 0)
        total_reward = 0
        path = [state]  # Record the initial state
        while state != (ROWS - 1, COLS - 1):
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = -1
            if next_state == (ROWS - 1, COLS - 1):
                reward = 10
            update_q_table(state, action, next_state, reward)
            state = next_state
            path.append(state)  # Record the visited state
            total_reward += reward
        rewards.append(total_reward)
        paths.append(path)  # Append the path for the current episode

        if episode % 100 == 0:
            visualize_path(path)  # Visualize the path for the current episode

    return rewards, paths

def main():
    rewards, paths = train()

    # Wait for user input to exit
    print("Training completed.")
    print("Press 'Enter' to exit or type 'q' and press 'Enter' to exit immediately.")
    user_input = input()
    if user_input.strip().lower() == 'q':
        return

if __name__ == "__main__":
    main()


