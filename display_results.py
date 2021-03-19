
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
    losses = np.array(list(data_generator(loss_path)))
    rewards = np.array(list(data_generator(rewards_path)))

    w = 1000
    losses = np.convolve(losses, np.ones(w)) / w
    rewards = np.convolve(rewards, np.ones(w)) / w

    fig, axs = plt.subplots(2)
    axs[0].plot(losses, color='r')
    axs[1].plot(rewards, color='b')
    plt.show()