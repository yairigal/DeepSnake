import sys

import cv2
import torch

from main import DeepQSnake
from main import SnakeBrain
from snake import ACTIONS
from snake import Snake

if __name__ == '__main__':
    model_file = sys.argv[1]

    model = SnakeBrain().double()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    s = Snake(6, 6)
    s.reset(food_amount=1)
    end = False
    cv2.namedWindow('snake', cv2.WINDOW_NORMAL)
    while not end:
        action = model(DeepQSnake._x(s)).argmax()
        end, r = s.step(action)
        cv2.imshow('snake', s.frames_queue[-1])
        cv2.waitKey(100)
        print(ACTIONS[action])
        print(f'r={r}, end={end}')