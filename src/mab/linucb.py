import numpy as np
import scipy as sp
from scipy import linalg

class LinUCB(object):

    def __init__(self, alpha = 0.25,
                 r1 = 0.8, r0 = 0,
                 d = 2, arms = []):
        self._alpha = alpha
        self._r1 = r1
        self._r0 = r0
        self._d = d
        self._arms = arms
        self._Aa = []
        self._AaI = []
        self._ba = []
        self._theta = []
        for arm in range(len(self._arms)):
            self._Aa.append(np.identity(d))
            self._ba.append(np.zeros((d, 1)))
            self._AaI.append(np.identity(d))
            self._theta.append(np.zeros((d, 1)))
        self._x = None
        self._xT = None

    def pull(self, arm):
        return int(np.random.rand() < self._arms[arm])

    def select(self, user_feature = None):
        if not user_feature:
            user_feature = np.identity(self._d)
        # context feature d x 1
        xaT = np.array([user_feature])
        xa = np.transpose(xaT)
        arm_count = len(self._arms)
        AaI_tmp = np.array([self._AaI[arm] for arm in range(arm_count)])
        theta_tmp = np.array([self._theta[arm] for arm in range(arm_count)])
        expected_reward = np.array([np.dot(xaT, self._theta[arm])
                                    for arm in range(arm_count)])
        bound = np.array([self._alpha * np.sqrt(np.dot(np.dot(xaT,
                                                              self._AaI[arm]),
                                                       xa))
                          for arm in range(arm_count)])
        confidence_bound = expected_reward + bound
        selected_arm = np.argmax(confidence_bound)

        self._x = xa
        self._xT = xaT
        return selected_arm

    def update(self, arm, reward):
        r = self._r1 if reward == 1 else self._r0
        self._Aa[arm] += np.dot(self._x, self._xT)
        self._ba[arm] += r * self._x
        self._AaI[arm] = linalg.solve(self._Aa[arm], np.identity(self._d))
        self._theta[arm] = np.dot(self._AaI[arm], self._ba[arm])

    def simulate(self, user):
        arm = self.select(user)
        reward = self.pull(arm)
        self.update(arm, reward)
        return arm


if __name__ == '__main__':
    arms = [0.8, 0.3]
    linucb = LinUCB(arms = arms)
    users = [[1,0], [0, 1]]
    t = 10
    for i in range(t):
        for user in users:
            arm = linucb.simulate(user)


