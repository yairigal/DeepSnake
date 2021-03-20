import argparse

import matplotlib.pyplot as plt
import numpy as np

loss_path = 'data/loss'
rewards_path = 'data/rewards'

def data_generator(path):
    with open(path, 'r') as f:
        while True:
            data = f.readline().strip('\n')
            if data == '':
                break

            if data != 'None':
                yield float(data)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=10, type=int, help='running average window size')
    args = parser.parse_args()


    losses = np.array(list(data_generator(loss_path)))
    rewards = np.array(list(data_generator(rewards_path)))

    w = args.weight
    losses = np.convolve(losses, np.ones(w)) / w
    rewards = np.convolve(rewards, np.ones(w)) / w

    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=3.0)
    axs[0].set_xlabel('episodes')
    axs[0].set_ylabel(f'loss')
    axs[0].plot(losses, color='r')
    axs[1].set_xlabel('episodes')
    axs[1].set_ylabel(f'rewards')
    axs[1].plot(rewards, color='b')
    plt.show()