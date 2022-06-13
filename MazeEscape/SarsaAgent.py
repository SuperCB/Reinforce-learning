import numpy as np


class SarsaAgent():
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        '''

        :param obs_n:
        :param act_n: 动作维度，有几个动作可选
        :param learning_rate: 学习率
        :param gamma: reward的衰减率
        :param e_greed: 按一定概率随机选动作
        '''

        self.act_n = act_n  #
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  #
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

        # Q表格的行是环境格子的总数，比如CliffWalking的Q表格有4*12=48行，FrozenLake的Q表格有4*4=16行。
        # Q结构是“环境格子数”*“动作数”。Q值即对应的每个格子采用某个动作所获得的终极回报

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        '''

        :param obs: 指的是当前状态
        :return:
        '''
        # if条件是根据状态选择能输出最大Q值的动作，有90%的机会执行exploitation
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:  # 有10%的机会执行exploration探索
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值，输出能产生最大Q值得action
    def predict(self, obs):
        '''

        :param obs: 指的是当前状态
        :return:
        '''


        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)  # 多个action则随机选一个
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy 即求target_Q时使用的self.Q[next_obs, next_action]是下一个obs-状态实际sample
                      到的action所对应的Q。即下一个obs和实际动作对应的Q
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]  # 获取对应单元格在动作action下的Q值
        if done:
            self.Q[obs, action] = self.Q[obs, action] + self.lr * (reward - predict_Q)
        else:
            self.Q[obs, action] = self.Q[obs, action] + self.lr * (
                    reward + self.gamma * self.Q[next_obs, next_action] - predict_Q)  # 修正q

    # # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取Q值到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
