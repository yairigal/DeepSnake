from collections import deque

import cv2
import numpy as np

ACTIONS = ['up', 'right', 'down', 'left']

# Game over causes
STARVATION, WALL, BODY, FULL = range(4)

VECTORS = [
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, -1])
]


class Snake:
    def __init__(self, h, w, food_amount=1):
        self.shape = (h, w)
        self.speed_vector = VECTORS[0]
        self.snake = np.array([[h // 2, w // 2]])
        self.food_amount = food_amount
        self.food = None
        self._spawn_food()
        self.max_steps = h * w * 2
        self.steps = 0
        self.game_over = False

        current_board = self._2d_board()
        self.frames_queue = deque([current_board, current_board], 2)

    def step(self, action):
        # no 180 degree turn
        if (VECTORS[action] + self.speed_vector).tolist() != [0, 0]:
            self.speed_vector = VECTORS[action]

        # Move head
        new_head = self.snake[0] + self.speed_vector

        h, w = self.shape
        x, y = new_head

        self.steps += 1

        # starvation
        if self.steps > self.max_steps:
            self.game_over = STARVATION
            return True

        # Collision with walls
        if x < 0 or x >= h or y < 0 or y >= w:
            self.game_over = WALL
            return True

        # Collision with body
        if new_head.tolist() in self.snake.tolist()[1:]:
            self.game_over = BODY
            return True

        # Collision with food
        if new_head.tolist() in self.food:
            self.food.remove(new_head.tolist())
            h, w = self.snake.shape
            s_h, s_w = self.shape
            self.snake = np.append(new_head, self.snake).reshape((h + 1, w))
            # snake is all the screen
            if h == s_h * s_w:
                self.game_over = FULL
                return True

            self._spawn_food()
            self.steps = 0

        else:
            self.snake = np.append(new_head, self.snake)[:-2].reshape(self.snake.shape)

        self.frames_queue.append(self._2d_board())
        return False

    @property
    def distance_from_food(self):
        return min(np.linalg.norm(food - self.snake[0]) for food in self.food)

    @property
    def score(self):
        return self.snake.shape[0] - 1

    def _3d_board(self):
        h, w = self.shape
        board = np.full((h, w, 3), 0)
        for i, c in enumerate((255, 255, 255)):
            board[:, :, i][tuple(self.snake.T)] = c

        for i, c in enumerate((255, 0, 0)):
            board[:, :, i][tuple(self.snake[0])] = c

        for food in self.food:
            for i, c in enumerate((0, 255, 0)):
                board[:, :, i][tuple(food)] = c
        # board[tuple(self.snake[0]), :] = (255, 0, 0)
        # board[tuple(self.food), :] = (0, 255, 0)
        return board / 255

    def _2d_board(self):
        h, w = self.shape
        board = np.full((h, w), 1, 'float')
        board[tuple(self.snake.T)] = 0.25

        board[tuple(self.snake[0])] = 0

        for food in self.food:
            board[tuple(food)] = 0.75
        return board

    @property
    def last_frames(self):
        return np.dstack(self.frames_queue)

    @property
    def board(self):
        return self.frames_queue[-1]

    def _spawn_food(self):
        h, w = self.shape
        if self.food is None:
            self.food = []

        while len(self.food) < self.food_amount:
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            blocks_left = (h * w) - len(self.food) - self.snake.shape[0]
            if blocks_left <= 0:
                return

            if [x, y] in self.snake.tolist() or [x, y] in self.food:
                continue

            self.food.append([x, y])


if __name__ == '__main__':
    s = Snake(6, 6, food_amount=5)
    s.max_steps = 1000
    moves = [
        'right',
        'right',
        'down',
        'down',
        'left',
        'left',
        'left',
        'left',
        'left',
        'up',
        'right',
        'right',
        'right',
        'right',
        'right',
        'up',
        'left',
        'left',
        'left',
        'left',
        'left',
        'up',
        'right',
        'right',
        'right',
        'right',
        'right',
        'up',
        'left',
        'left',
        'left',
        'left',
        'left',
        'up',
        'right',
        'right',
        'right',
        'right',
        'right',
    ]
    cv2.namedWindow('snake', cv2.WINDOW_NORMAL)
    for move in moves:
        end = s.step(ACTIONS.index(move))
        cv2.imshow('snake', s.board)
        cv2.waitKey(100)
        print(f'move={move}, end={end}')
