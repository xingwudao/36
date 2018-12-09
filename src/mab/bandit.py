import numpy as np
from numpy.random import beta
import matplotlib.pyplot as plt

class Bandit(object):

    def __init__(self, arm_priority):
        self._cumulative_regret_list = []
        self._cumulative_regret = 0
        self._priority = arm_priority
        self._best= np.max(self._priority)
        self._win_of_arms = np.zeros(len(arm_priority))
        self._loss_of_arms = np.zeros(len(arm_priority))

    def pull(self, arm):
        return int(np.random.rand() < self._priority[arm])

    def select(self):
        raise Exception('not implement')

    def update(self, arm, reward):
        self._win_of_arms[arm] += reward
        self._loss_of_arms[arm] += 1 - reward
        regret = self._best - self._priority[arm]
        self._cumulative_regret += regret
        self._cumulative_regret_list.append(
            self._cumulative_regret)

    def simulate(self):
        arm = self.select()
        reward = self.pull(arm)
        self.update(arm, reward)
    
    @property
    def cumulative_regret(self):
        return self._cumulative_regret

    @property
    def cumulative_regret_list(self):
        return self._cumulative_regret_list

class ThompsonSamplingBandit(Bandit):

    def __init__(self, arm_priority):
        Bandit.__init__(self, arm_priority)

    def select(self):
        randoms = beta(1+self._win_of_arms, 1+self._loss_of_arms)
        return np.argmax(randoms)

class UCBBandit(Bandit):

    def __init__(self, arm_priority):
        Bandit.__init__(self, arm_priority)
        self._trials = 0
        self._avg_reward = np.zeros(len(arm_priority))

    def select(self):
        trial_of_arms = self._win_of_arms + self._loss_of_arms
        avg = self._avg_reward
        avg += np.sqrt(2*np.log(1+self._trials)/(1+trial_of_arms)) 
        return np.argmax(avg)

    def update(self, arm, reward):
        Bandit.update(self, arm, reward)
        self._trials += 1
        trials_of_arm = self._win_of_arms[arm] + self._loss_of_arms[arm]
        self._avg_reward[arm] = ((trials_of_arm - 1)*self._avg_reward[arm]
                            + self._win_of_arms[arm])/trials_of_arm


class EpsilonGreedyBandit(Bandit):

    def __init__(self, arm_priority, epsilon, min_trials = 0):
        Bandit.__init__(self, arm_priority)
        self._epsilon = epsilon
        self._avg_reward = np.zeros(len(arm_priority))
        self._trials = 0
        self._min_trials =  min_trials

    def select(self):
        if (np.random.rand() < self._epsilon
            or self._trials <  self._min_trials):
            return np.random.choice(range(len(self._win_of_arms)))
        arm = np.argmax(self._avg_reward)
        return arm

    def update(self, arm, reward):
        Bandit.update(self, arm, reward)
        self._trials += 1
        trials_of_arm = self._win_of_arms[arm] + self._loss_of_arms[arm]
        self._avg_reward[arm] = self._win_of_arms[arm]/trials_of_arm
#        self._avg_reward[arm] = ((trials_of_arm - 1)*self._avg_reward[arm]
#                            + self._win_of_arms[arm])/trials_of_arm
    

if __name__ == '__main__':
    priority = [0.15, 0.20, 0.42]
    name1, bandit_1 = "ThompsonSampling", ThompsonSamplingBandit(priority)
    name2, bandit_2 = "UCB", UCBBandit(priority)
    name3, bandit_3 = "greedy", EpsilonGreedyBandit(priority, 0)
    name5, bandit_5 = "epsilon0.05", EpsilonGreedyBandit(priority, 0.05)
    name8, bandit_8 = "random", EpsilonGreedyBandit(priority, 1)
    name9, bandit_9 = "greedy-naive", EpsilonGreedyBandit(priority, 3)
    t = 1000
    for i in range(t):
        bandit_1.simulate()
        bandit_2.simulate()
        bandit_3.simulate()
        bandit_5.simulate()
        bandit_8.simulate()
        bandit_9.simulate()
   
    c1, = plt.plot(range(t), bandit_1.cumulative_regret_list)
    c2, = plt.plot(range(t), bandit_2.cumulative_regret_list)
    c3, = plt.plot(range(t), bandit_3.cumulative_regret_list)
    c5, = plt.plot(range(t), bandit_5.cumulative_regret_list)
    c8, = plt.plot(range(t), bandit_8.cumulative_regret_list)
    c9, = plt.plot(range(t), bandit_9.cumulative_regret_list)

    plt.ylabel('cumulative regret')
    plt.xlabel('t')
    plt.legend(handles= [c1, c2, c3, c5,  c8, c9],
               labels = [name1, name2, name3, name5, name8, name9],
               loc = 'best')
    plt.show()
