import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
import pygame

# Constants for DQN and Tic-Tac-Toe
BOARD_SIZE = 3
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.95
LEARNING_RATE = 0.001
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# Initialize the Tic-Tac-Toe environment
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

# Create the DQN agent
class DQNAgent:
    def __init__(self, empty_value, player_x, player_o):
        self.model = self.build_dqn_model(input_shape=(NUM_ACTIONS,), num_actions=NUM_ACTIONS)
        self.target_model = self.build_dqn_model(input_shape=(NUM_ACTIONS,), num_actions=NUM_ACTIONS)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
        self.EMPTY = empty_value
        self.PLAYER_X = player_x
        self.PLAYER_O = player_o

    def build_dqn_model(self, input_shape, num_actions):
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
        return model

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

# Initialize the environment and agent
env = TicTacToeEnvironment()
agent = DQNAgent(EMPTY, PLAYER_X, PLAYER_O)

# Initialize pygame
pygame.init()
CELL_SIZE = 100
screen = pygame.display.set_mode((BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
pygame.display.set_caption("Tic Tac Toe")


# Functions for gameplay
def get_random_move(board):
    open_spaces = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                open_spaces.append((row, col))
    if open_spaces:
        return random.choice(open_spaces)
    else:
        return ()

def is_winner(board, player):
    for row in range(BOARD_SIZE):
        if all(board[row][col] == player for col in range(BOARD_SIZE)):
            return True
    for col in range(BOARD_SIZE):
        if all(board[row][col] == player for row in range(BOARD_SIZE)):
            return True
    if all(board[row][row] == player for row in range(BOARD_SIZE)):
        return True
    if all(board[row][BOARD_SIZE - row - 1] == player for row in range(BOARD_SIZE)):
        return True
    return False

def draw(board):
    screen.fill((255, 255, 255))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, (0, 0, 0), (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2)
            if board[row][col] == agent.EMPTY:
                continue
            elif board[row][col] == agent.PLAYER_X:
                pygame.draw.line(screen, (255, 0, 0), (col * CELL_SIZE, row * CELL_SIZE),
                                 ((col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE), 2)
                pygame.draw.line(screen, (255, 0, 0), (col * CELL_SIZE, (row + 1) * CELL_SIZE),
                                 ((col + 1) * CELL_SIZE, row * CELL_SIZE), 2)
            elif board[row][col] == agent.PLAYER_O:
                pygame.draw.circle(screen, (0, 0, 255),
                                   (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 2 - 5, 2)

def handle_input(board):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            row = mouse_pos[1] // CELL_SIZE
            col = mouse_pos[0] // CELL_SIZE
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == agent.EMPTY:
                return row, col
    return None

# Main loop for gameplay
def main():
    while True:
        draw(env.board)
        pygame.display.update()
        pygame.time.delay(100)

        # Check for winner
        if is_winner(env.board, agent.PLAYER_X):
            print("Player X wins!")
            break
        elif is_winner(env.board, agent.PLAYER_O):
            print("Player O wins!")
            break

        # Check for a draw
        if all(space != EMPTY for row in env.board for space in row):
            print("Draw!")
            break

        if env.current_player == agent.PLAYER_X:
            # Player's turn
            player_move = handle_input(env.board)
            if player_move is not None:
                row, col = player_move
                env.board[row][col] = agent.PLAYER_X
                env.current_player = agent.PLAYER_O
        else:
            # Agent's turn
            agent_move = get_random_move(env.board)
            if agent_move:
                row, col = agent_move
                env.board[row][col] = agent.PLAYER_O
                env.current_player = agent.PLAYER_X

        agent.train()

# Run the main loop
if __name__ == "__main__":
    main()
