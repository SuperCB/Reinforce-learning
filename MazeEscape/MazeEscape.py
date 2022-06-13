import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
from copy import deepcopy
import numpy as np
import pygame

from MazeEscape.QlearningAgent import QLearningAgent
from MazeEscape.SarsaAgent import SarsaAgent


class Player():
    def __init__(x, y, self):
        self.x = x
        self.y = y


class MazeEscape(gym.Env):
    #

    def __init__(self, agent=None):
        self.map = []
        self.player = []

        with open('map.txt', 'r') as file:
            data = file.read().splitlines()  # 这种读法能去掉换行符
            width, height = data[0].split(' ')
            bx, by = data[1].split(' ')
            self.bx = eval(bx)
            self.by = eval(by)

            self.width = eval(width)
            self.height = eval(height)

            for i in range(2, len(data)):
                self.map.append(data[i].split(' '))

        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.width * self.height)

        wall = pygame.image.load('images\\wall.png')
        self.wall = pygame.transform.scale(wall, (50, 50))

        grass = pygame.image.load('images\\grass.png')
        self.grass = pygame.transform.scale(grass, (50, 50))

        player = pygame.image.load('images\\player.png')
        self.player = pygame.transform.scale(player, (50, 50))

        coin = pygame.image.load('images\\door.png')
        self.coin = pygame.transform.scale(coin, (50, 50))

    def check(self):
        pass

    def step(self, action):
        # new_state x,y
        new_state = deepcopy(self.current_state)

        if action == 0:  # right
            new_state[0] = min(new_state[0] + 1, self.width - 1)

        elif action == 1:  # down
            new_state[1] = min(new_state[1] + 1, self.height - 1)

        elif action == 2:  # left
            new_state[0] = max(new_state[0] - 1, 0)

        elif action == 3:  # up
            new_state[1] = max(new_state[1] - 1, 0)
        else:
            raise Exception("Invalid action.")

        is_terminal = False
        reward = None

        label = self.map[new_state[1]][new_state[0]]
        if label == 'w':
            reward = -1.0
        elif label == 'g':
            reward = 0
            self.current_state = new_state
        elif label == 'c':
            reward = 100
            is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def observation(self, state):
        return state[1] * self.width + state[0]

    def sample(self):
        '''
        建议不使用这个函数
        :return:
        '''

        actions = [0, 1, 2, 3]
        return np.random.choice(actions, 1, replace=False)[0]

    def render(self, mode="human"):
        pygame.init()

        self.background = pygame.display.set_mode((self.width * 50, self.height * 50))

        pygame.display.set_caption('Escape from the maze')

        clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pass
            elif event.type == pygame.KEYDOWN:
                pass

        clock.tick(10)  # 每秒循v环60次
        for i in range(self.height):
            for j in range(self.width):
                self.background.blit(self.grass, (j * 50, i * 50))
                if self.map[i][j] == 'c':
                    self.background.blit(self.coin, (j * 50, i * 50))
                elif self.map[i][j] == 'w':
                    self.background.blit(self.wall, (j * 50, i * 50))
        self.background.blit(self.player, (self.current_state[0] * 50, self.current_state[1] * 50))
        pygame.display.update()
        # time.sleep(0.1)

    def reset(self):
        self.current_state = [self.bx, self.by]
        return self.observation(self.current_state)


def run_episode(env, agent, render=False):
    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互，由action->state与策略无关，
        # 由环境决定，由环境的状态转移概率决定。
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作，
        # 由策略决定，一般平衡exploration和exploitation
        # 训练 Sarsa 算法，更新Q值表
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action  # 该动作是下次实际采用动作
        obs = next_obs  # 存储上一个观察值
        env.render()
        if done:
            break


if __name__ == "__main__":
    env = MazeEscape()
    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    action = env.sample()  # 根据算法选择一个动作
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    # while True:
    #     next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互，由action->state与策略无关，
    #     # 由环境决定，由环境的状态转移概率决定。
    #     next_action = agent.sample(next_obs)  # 根据算法选择一个动作，
    #     # 由策略决定，一般平衡exploration和exploitation
    #     # 训练 Sarsa 算法，更新Q值表
    #     agent.learn(obs, action, reward, next_obs, done)
    #
    #     action = next_action  # 该动作是下次实际采用动作
    #     obs = next_obs  # 存储上一个观察值
    #     env.render()  # 渲染新的一帧图形
    #     if done:
    #         break

    for episode in range(500):
        print('Episode %r\n' % episode)

        run_episode(env, agent, True)

    agent.save()
