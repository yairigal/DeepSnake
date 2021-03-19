import time
from collections import deque

import numpy as np
import cv2

ACTIONS = ['up', 'right', 'down', 'left']

VECTORS = [
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, -1])
]


class Snake:
    def __init__(self, h, w):
        self.shape = (h, w)
        self.state_space = (h, w, 2)
        self.speed_vector = None
        self.snake = None
        self.food = None
        self.action_space = len(ACTIONS)
        self.max_steps = h * w * 2
        self.steps = None
        self.food_amount = None
        self.frames_queue = deque([], 2)
        self.game_over = False
        self.reset()

    def reset(self, food_amount=1):
        h, w = self.shape
        self.speed_vector = [0]
        self.snake = np.array([[h // 2, w // 2]])
        self.food_amount = food_amount
        self.food = None
        self.food = self._spawn_food()
        # self.food = np.array([5,5])
        self.steps = 0
        self.frames_queue.append(self._2d_board())
        self.frames_queue.append(self._2d_board())
        return self.get_state()

    def step(self, action):
        # no 180 degree turn
        if (VECTORS[action] + self.speed_vector).tolist() != [0, 0]:
            self.speed_vector = VECTORS[action]

        new_head = self.snake[0] + self.speed_vector

        h, w = self.shape
        x, y = new_head

        self.steps += 1

        # starvation
        if self.steps > self.max_steps:
            self.game_over = True
            return True, 0

        # Collision with walls
        if x < 0 or x >= h or y < 0 or y >= w:
            self.game_over = True
            return True, -1

        # Collision with body
        if new_head.tolist() in self.snake.tolist()[1:]:
            self.game_over = True
            return True, -1

        # Collision with food
        if new_head.tolist() in self.food:
            self.food.remove(new_head.tolist())
            h, w = self.snake.shape
            self.snake = np.append(new_head, self.snake).reshape((h + 1, w))
            s_h, s_w = self.shape
            # snake is all the screen
            if self.snake.shape[0] - 1 == s_h * s_w:
                self.game_over = True
                return True

            self.food = self._spawn_food()
            self.steps = 0
            self.frames_queue.append(self._2d_board())
            return False, 1

        self.snake = np.append(new_head, self.snake)[:-2].reshape(self.snake.shape)
        self.frames_queue.append(self._2d_board())
        return False, 0

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
        # board[tuple(self.snake[0]), :] = (255, 0, 0)
        # board[tuple(self.food), :] = (0, 255, 0)
        return board

    def get_state(self):
        return np.dstack(self.frames_queue)

    def _spawn_food(self):
        h, w = self.shape
        foods = 0
        f = []
        if self.food is not None:
            foods = len(self.food)
            f = self.food

        while foods < self.food_amount:
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            if [x, y] in self.snake.tolist() or [x, y] in f:
                continue

            foods += 1
            f.append([x, y])

        return f


if __name__ == '__main__':
    s = Snake(6, 6)
    s.reset(food_amount=1)
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
        end, r= s.step(ACTIONS.index(move))
        cv2.imshow('snake', s.frames_queue[-1])
        cv2.waitKey(100)
        print(move)
        print(f'r={r}, end={end}')