import random
import pygame

# Constants
BOARD_SIZE = 3
CELL_SIZE = 100
PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
pygame.display.set_caption("Tic Tac Toe")

# Functions
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
    # Check for winner logic
    # Return True if player has won

    for row in range(BOARD_SIZE):
        if all(board[row][col] == player for col in range(BOARD_SIZE)):
            return True


    for col in range(BOARD_SIZE):
        if all(board[row][col] == player for row in range(BOARD_SIZE)):
            return True

    if all(board[row][row] == player for row in range(BOARD_SIZE)):  #top-left corner
        return True

    if all(board[row][BOARD_SIZE - row - 1] == player for row in range(BOARD_SIZE)):
        return True

    # The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.

    return False


def draw(board):
    screen.fill((255, 255, 255))  # Fill the screen with white
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, (0, 0, 0), (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2)
            if board[row][col] == PLAYER_X:
                pygame.draw.line(screen, (255, 0, 0), (col * CELL_SIZE, row * CELL_SIZE),
                                 ((col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE), 2)
                pygame.draw.line(screen, (255, 0, 0), (col * CELL_SIZE, (row + 1) * CELL_SIZE),
                                 ((col + 1) * CELL_SIZE, row * CELL_SIZE), 2)
            elif board[row][col] == PLAYER_O:
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
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == EMPTY:
                return row, col
    return None  # No valid cell clicked

def main():
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = PLAYER_X

    while True:
        draw(board)

        # Check for winner
        if is_winner(board, PLAYER_X):
            print("Player X wins!")
            break
        elif is_winner(board, PLAYER_O):
            print("Player O wins!")
            break

        # Check for a draw
        if all(space != EMPTY for row in board for space in row):
            print("Draw!")
            break

        if current_player == PLAYER_X:
            # Player's turn
            player_move = handle_input(board)
            if player_move is not None:
                row, col = player_move
                board[row][col] = current_player
                current_player = -current_player
        else:
            # Agent's turn
            agent_move = get_random_move(board)
            if agent_move is not None:
                row, col = agent_move
                board[row][col] = current_player
                current_player = -current_player

        pygame.display.update()
        pygame.time.delay(100)

if __name__ == "__main__":
    main()