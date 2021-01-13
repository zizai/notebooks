import os
from datetime import time

import gym
import matplotlib.pyplot as plt
import torch
from torch.distributions import OneHotCategorical

from neuroblast.agents.diayn import DIAYN


def mountaincar(diayn, filename):
    n = 1
    plt.figure()
    n_skills = diayn.prior.event_shape[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for skill in range(n_skills):
        color = colors[skill]
        for i in range(n):
            z = torch.zeros((1, diayn.prior.event_shape[0]))
            z[0, skill] = 1
            states = diayn.episode(train=False, z=z, return_states=True)
            positions = list(zip(*states))[0]
            kwargs = {"color": color, "alpha": .3}
            if i == 0:
                kwargs["label"] = "Skill n° {}".format(skill)
            plt.plot(positions, **kwargs)
    plt.xlabel("Step")
    plt.ylabel("$x$ position")
    plt.legend()
    plt.show()
    plt.pause(1)
    plt.savefig(filename)
    plt.close()


def main(n_skills, path):
    '''
    :param n_skills:
    :return:
    '''
    env = gym.make('MountainCar-v0')
    alpha = .1
    gamma = .9
    prior = OneHotCategorical(torch.ones((1, n_skills)))
    hidden_sizes = {s: [30,30] for s in ("actor", "discriminator", "critic")}
    trainer = DIAYN(env, prior, hidden_sizes, alpha=alpha, gamma=gamma)

    path_plot = path + "plot_diayn\\"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    path_save = path + "save_diayn\\"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for k in range(0, 1):
        iter_ = 200
        trainer.train(iter_)
        trainer.plot_rewards(path_plot + "diyan_train_rewards_" + str((k+1)*iter_))
        #input("Press Enter to see skills")
        mountaincar(trainer, path_plot + "diyan_train_trajectoires_" + str((k+1)*iter_))
        # plt.show() not needed since plt.ion() is called in diayn.py
        # plt.pause(1)
        trainer.save(path_save)


if __name__ == '__main__':
    main(10, '/tmp/diayn_{}'.format(time()))
