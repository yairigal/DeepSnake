import sys

import cv2
import torch

from main import DeepQSnake
from main import SnakeBrain
from snake import ACTIONS
from snake import Snake
import numpy as np
import imageio

if __name__ == '__main__':
    model_file = sys.argv[1]

    model = SnakeBrain().double()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    s = Snake(6, 6)
    s.reset(food_amount=1)
    end = False
    cv2.namedWindow('snake', cv2.WINDOW_NORMAL)
    images = []
    while not end:
        action = model(DeepQSnake._x(s)).argmax()
        end, r = s.step(action)
        frame = s.frames_queue[-1]
        frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('snake', frame)
        images.append(np.uint8(frame * 255))
        cv2.waitKey(100)
        print(ACTIONS[action])
        print(f'r={r}, end={end}')

    imageio.mimsave('snake.gif', images)