import os
import json
import random
import logging
import argparse
import datetime
from copy import deepcopy
from collections import deque
from collections import namedtuple

import torch
import numpy as np
from torch.nn import MSELoss
from torch.optim import RMSprop
import torch.nn.functional as F

import snake
from snake import Snake
from snake import ACTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s >> %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

Replay = namedtuple('replay', ['s', 'a', 's_t', 'r', 'end'])

class SnakeBrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 16, (3, 3))  # 6x6x2 => 4x4x16
        self.conv2 = torch.nn.Conv2d(16, 32, (3, 3))  # 4x4x16 => 2x2x32
        self.fc1 = torch.nn.Linear(32 * 2 * 2, 64)
        self.fc2 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO two networks
class DeepQSnake:
    save_path = 'checkpoints'

    def __init__(self,
                 board_size,
                 e=1,
                 e_decay=.97,
                 e_min=0.1,
                 e_decay_frequency=1000,
                 batch_size=64,
                 max_replays=200_000,
                 episodes=1_000_000,
                 gamma=.99,
                 learning_rate=0.0005,
                 save_frequency=10000,
                 print_frequency=1000,
                 games_per_episode=8,
                 path_to_model=None):
        self.snake_shape = board_size
        self.e = e
        self.e_decay = e_decay
        self.e_min = e_min
        self.decay_frequency = e_decay_frequency
        self.batch_size = batch_size
        self.max_replays = max_replays
        self.episodes = episodes
        self.gamma = gamma
        self.LR = learning_rate
        self.save_frequency = save_frequency
        self.print_frequency = print_frequency
        self.games = games_per_episode
        self.replays = deque([], self.max_replays)
        self.frames_played = 0
        self.losses_file = open('data/loss', 'w')
        self.rewards_file = open('data/rewards', 'w')

        self.network = SnakeBrain().double()
        if path_to_model is not None:
            logger.debug(f'loaded model from {path_to_model}')
            self.network.load_state_dict(torch.load(path_to_model))

        self.optimizer = RMSprop(self.network.parameters(), lr=self.LR)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    @staticmethod
    def _x(env):
        x = env.last_frames
        h, w, c = x.shape
        x = x.reshape(1, c, h, w)
        x = np.double(x)
        return torch.tensor(x)

    # TODO CHECK TENSORFLOW
    def next_action(self, env: Snake):
        if random.random() <= self.e:
            return random.randint(0, len(ACTIONS) - 1)

        return self.network(self._x(env)).argmax()

    def reward(self, s: Snake, s_t: Snake):
        """Calculate reward for that input transition.

        Args:
            s (Snake): current state.
            s_t (Snake): next state.

        Returns:
            number. the reward for that phase.
        """
        if s_t.game_over and s_t.game_over != snake.FULL:
            assert s.game_over != s_t.game_over
            return -1

        if s_t.score > s.score:
            return 1

        return 0


    def episode(self):
        env = Snake(*self.snake_shape)
        end = False
        total_rewards = 0
        while not end:
            a = self.next_action(env)
            s = deepcopy(env)
            end = env.step(a)
            s_t = deepcopy(env)
            r = self.reward(s, s_t)
            self.frames_played += 1
            replay = Replay(s=s, s_t=s_t, r=r, a=a, end=end)
            self.replays.append(replay)
            total_rewards += r

        return total_rewards

    def _predict_on_batch(self, env_list):
        X = torch.cat([self._x(s) for s in env_list])
        return self.network(X)

    def _train(self):
        batch = random.sample(self.replays, self.batch_size)
        S, A, S_T, R, END = list(zip(*batch))
        R = torch.tensor(R)  # m, 1
        END = torch.tensor(END, dtype=torch.int)
        A = torch.nn.functional.one_hot(torch.tensor(A), num_classes=4)  # m, 4

        y_s_t = self._predict_on_batch(S_T)  # m, 4
        assert y_s_t.shape == (self.batch_size, len(ACTIONS))

        q_t = y_s_t.max(axis=1).values
        assert q_t.shape == (self.batch_size,)

        discounted_reward = R + self.gamma * q_t * (1 - END)

        target = self._predict_on_batch(S)
        target = target * (1 - A) + discounted_reward.reshape(self.batch_size, 1) * A

        # loss = MSELoss(reduction='mean')(target.detach(), self._predict_on_batch(S))
        loss = F.smooth_l1_loss(target.detach(), self._predict_on_batch(S))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return float(loss)

    def train(self):
        total_rewards = 0
        last_avg = 0
        loss = None
        for ep in range(self.episodes):
            total_rewards += sum(self.episode() for _ in range(self.games))

            if (ep + 1) % self.decay_frequency == 0:
                self.e = max(self.e * self.e_decay, self.e_min)

            if len(self.replays) >= self.batch_size:
                loss = self._train()
                self.losses_file.write(str(loss) + '\n')
                self.rewards_file.write(str(total_rewards / self.games) + '\n')
                self.losses_file.flush()
                self.rewards_file.flush()
                last_avg = total_rewards / self.games
                total_rewards = 0

            if ep % self.print_frequency == 0:
                logger.debug(f"\n"
                             f"    episode #{ep + 1}\n"
                             f"    frames played: {self.frames_played}\n"
                             f"    loss: {loss}\n"
                             f"    average_rewards: {last_avg}\n"
                             f"    replays: {len(self.replays)}\n"
                             f"    e: {self.e}")

            if (ep + 1) % self.save_frequency == 0:
                save_name = datetime.datetime.now().isoformat(sep='_', timespec='minutes').replace(":", "_") + "_" + \
                            str(ep + 1)
                logger.debug(f"Saving model at {save_name}")
                torch.save(self.network.state_dict(), os.path.join(self.save_path, save_name))

        self.losses_file.close()
        self.rewards_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default=None, help='path for old checkpoint to train from')
    parser.add_argument("--config", default='config.json', help='path for config file')
    args = parser.parse_args()
    with open(args.config, 'r') as c:
        config = json.load(c)

    DeepQSnake((6, 6), path_to_model=args.load, **config).train()
