import os

import numpy as np
import skimage

import cfg
import argparse
from modules.sprites.Pipe import *
from modules.sprites.Bird import *
from modules.interfaces.endGame import *
from modules.interfaces.startGame import *
from Algorithms.Agents.dqn_agent import *
from Algorithms.Networks.dqn_cnn import *
import skimage.color
import skimage.exposure
import skimage.transform


class FlappyBird():

    def __init__(self):
        pass

    def reset(self):
        pygame.init()
        # the music
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
        pygame.display.set_caption('FlappyBird')
        self.clock = pygame.time.Clock()

        self.score = 0.0
        self.sounds = dict()
        for key, value in cfg.AUDIO_PATHS.items():
            self.sounds[key] = pygame.mixer.Sound(value)

        self.number_images = dict()
        for key, value in cfg.NUMBER_IMAGE_PATHS.items():
            self.number_images[key] = pygame.image.load(value).convert_alpha()

        self.pipe_images = dict()
        self.pipe_images['bottom'] = pygame.image.load(
            random.choice(list(cfg.PIPE_IMAGE_PATHS.values()))).convert_alpha()
        self.pipe_images['top'] = pygame.transform.rotate(self.pipe_images['bottom'], 180)

        self.bird_images = dict()
        for key, value in cfg.BIRD_IMAGE_PATHS[random.choice(list(cfg.BIRD_IMAGE_PATHS.keys()))].items():
            self.bird_images[key] = pygame.image.load(value).convert_alpha()

        self.backgroud_image = pygame.image.load(
            random.choice(list(cfg.BACKGROUND_IMAGE_PATHS.values()))).convert_alpha()

        self.other_images = dict()
        for key, value in cfg.OTHER_IMAGE_PATHS.items():
            self.other_images[key] = pygame.image.load(value).convert_alpha()

        bird_idx = 0
        bird_pos = [cfg.SCREENWIDTH * 0.2, cfg.SCREENHEIGHT * 0.5]
        self.bird = Bird(images=self.bird_images, idx=bird_idx, position=bird_pos)

        self.pipe_sprites = pygame.sprite.Group()
        self.base_pos = [0, cfg.SCREENHEIGHT * 0.79]
        self.boundary_values = [0, self.base_pos[-1]]

        self.base_diff_bg = self.other_images['base'].get_width() - self.backgroud_image.get_width()

        for i in range(2):
            pipe_pos = Pipe.randomPipe(cfg, self.pipe_images.get('top'))
            self.pipe_sprites.add(Pipe(image=self.pipe_images.get('top'),
                                       position=(
                                           cfg.SCREENWIDTH + 200 + i * cfg.SCREENWIDTH / 2, pipe_pos.get('top')[-1]),
                                       type_='top'))
            self.pipe_sprites.add(Pipe(image=self.pipe_images.get('bottom'),
                                       position=(
                                           cfg.SCREENWIDTH + 200 + i * cfg.SCREENWIDTH / 2, pipe_pos.get('bottom')[-1]),
                                       type_='bottom'))

        self.is_add_pipe = True

        state = pygame.surfarray.array3d(pygame.display.get_surface())

        # the type of image is numpy ndarray

        state = state[:, :int(0.79 * cfg.SCREENHEIGHT), :]

        return self.preprocess(state)

    def endgame(self):
        '''
        游戏结束时调用的函数
        :return:
        '''
        self.sounds['die'].play()
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        return
            boundary_values = [0, self.base_pos[-1]]
            self.bird.update(boundary_values)
            self.screen.blit(self.backgroud_image, (0, 0))
            self.pipe_sprites.draw(self.screen)
            self.screen.blit(self.other_images['base'], self.base_pos)
            self.showScore(self.score)
            self.bird.draw(self.screen)
            pygame.display.update()
            clock.tick(cfg.FPS)

    def showScore(self, score):
        digits = list(str(int(score)))
        width = 0
        for d in digits:
            width += self.number_images.get(d).get_width()
        offset = (cfg.SCREENWIDTH - width) / 2
        for d in digits:
            self.screen.blit(self.number_images.get(d), (offset, cfg.SCREENHEIGHT * 0.1))
            offset += self.number_images.get(d).get_width()

    # next_state, reward, done, info = env.step(action)

    def step(self, action):

        done = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        reward = 0.1

        if action:
            self.bird.setFlapped()
            self.sounds['wing'].play()

        # --check the collide between bird and pipe
        for pipe in self.pipe_sprites:
            if pygame.sprite.collide_mask(self.bird, pipe):
                self.sounds['hit'].play()
                done = True
                reward = -1

        # --update the bird
        is_dead = self.bird.update(self.boundary_values)

        if is_dead:
            self.sounds['hit'].play()
            done = True
            reward = -1

        # --move the bases to the left to achieve the effect of bird flying forward

        self.base_pos[0] = -((-self.base_pos[0] + 4) % self.base_diff_bg)
        # --move the pipes to the left to achieve the effect of bird flying forward

        for pipe in self.pipe_sprites:
            pipe.rect.left -= 4
            if pipe.rect.centerx <= self.bird.rect.centerx and not pipe.used_for_score:
                pipe.used_for_score = True
                self.score += 0.5
                reward = 1
                if '.5' in str(self.score):
                    self.sounds['point'].play()

            if pipe.rect.left < 5 and pipe.rect.left > 0 and self.is_add_pipe:
                pipe_pos = Pipe.randomPipe(cfg, self.pipe_images.get('top'))
                self.pipe_sprites.add(
                    Pipe(image=self.pipe_images.get('top'), position=pipe_pos.get('top'), type_='top'))
                self.pipe_sprites.add(
                    Pipe(image=self.pipe_images.get('bottom'), position=pipe_pos.get('bottom'), type_='bottom'))
                self.is_add_pipe = False
            elif pipe.rect.right < 0:
                self.pipe_sprites.remove(pipe)
                self.is_add_pipe = True

        self.pipe_sprites.draw(self.screen)
        self.bird.draw(self.screen)

        state = pygame.surfarray.array3d(pygame.display.get_surface())

        # the type of image is numpy ndarray

        next_state = state[:, :int(0.79 * cfg.SCREENHEIGHT), :]

        # --blit the necessary game elements on the screen

        self.screen.blit(self.backgroud_image, (0, 0))
        self.pipe_sprites.draw(self.screen)
        self.screen.blit(self.other_images['base'], self.base_pos)
        self.bird.draw(self.screen)
        # --record the action and corresponding reward

        pygame.display.update()
        self.clock.tick(cfg.FPS)

        return self.preprocess(next_state), reward, done, ' '

    def sample(self):
        return 0

    def preprocess(self, image):

        # image = skimage.color.rgb2gray(image)
        image = skimage.transform.resize(image, (100, 100), mode='constant')
        image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        return image


# def train(n_episodes=1000):
#     """
#     Params
#     ======
#         n_episodes (int): maximum number of training episodes
#     """
#     for i_episode in range(start_epoch + 1, n_episodes + 1):
#         state = stack_frames(None, env.reset(), True)
#         score = 0
#         eps = epsilon_by_epsiode(i_episode)
#         while True:
#             action = agent.act(state, eps)
#             next_state, reward, done, info = env.step(action)
#             score += reward
#             next_state = stack_frames(state, next_state, False)
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 break
#         scores_window.append(score)  # save most recent score
#         scores.append(score)  # save most recent score
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
#
#         if i_episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             plt.plot(np.arange(len(scores)), scores)
#             plt.ylabel('Score')
#             plt.xlabel('Episode #')
#             plt.show()
#
#     return scores
if __name__ == '__main__':
    # modelpath = 'checkpoints/dqn.pth'
    INPUT_SHAPE = (3, 100, 100)
    ACTION_SIZE = 2
    SEED = 0
    GAMMA = 0.99  # discount factor
    BUFFER_SIZE = 100000  # replay buffer size
    BATCH_SIZE = 64  # Update batch size
    LR = 0.0001  # learning rate
    TAU = 1e-3  # for soft update of target parameters
    UPDATE_EVERY = 1  # how often to update the network
    UPDATE_TARGET = 10000  # After which thershold replay to be started
    EPS_START = 0.99  # starting value of epsilon
    EPS_END = 0.01  # Ending value of epsilon
    EPS_DECAY = 100  # Rate by which epsilon to be decayed
    device = 'cuda'

    agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY,
                     UPDATE_TARGET, DQNCnn)

    env = FlappyBird()

    score = 0
    for i in range(100):
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            # next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    # if i_episode % 100 == 0:
    #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     plt.plot(np.arange(len(scores)), scores)
    #     plt.ylabel('Score')
    #     plt.xlabel('Episode #')
    #     plt.show()
