import os
import cfg
import argparse
from modules.sprites.Pipe import *
from modules.sprites.Bird import *
from modules.interfaces.endGame import *
from modules.interfaces.startGame import *
from modules.QLearningAgent.QLearningAgent import *


class FlappyBird():
    def __int__(self):
        self.sounds = dict()
        for key, value in cfg.AUDIO_PATHS.items():
            self.sounds[key] = pygame.mixer.Sound(value)

        self.number_images = dict()
        for key, value in cfg.NUMBER_IMAGE_PATHS.items():
            self.number_images[key] = pygame.image.load(value).convert_alpha()

        self.pipe_images = dict()
        self.pipe_images['bottom'] = pygame.image.load(
            random.choice(list(cfg.PIPE_IMAGE_PATHS.values()))).convert_alpha()
        self.pipe_images['top'] = pygame.transform.rotate(pipe_images['bottom'], 180)

        self.bird_images = dict()
        for key, value in cfg.BIRD_IMAGE_PATHS[random.choice(list(cfg.BIRD_IMAGE_PATHS.keys()))].items():
            self.bird_images[key] = pygame.image.load(value).convert_alpha()

        self.backgroud_image = pygame.image.load(
            random.choice(list(cfg.BACKGROUND_IMAGE_PATHS.values()))).convert_alpha()

        self.other_images = dict()
        for key, value in cfg.OTHER_IMAGE_PATHS.items():
            self.other_images[key] = pygame.image.load(value).convert_alpha()

    def reset(self):
        pygame.init()
        # the music
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
        pygame.display.set_caption('FlappyBird')

    def startgame(self):

        base_pos = [0, cfg.SCREENHEIGHT * 0.79]
        base_diff_bg = self.other_images['base'].get_width() - self.backgroud_image.get_width()
        msg_pos = [(cfg.SCREENWIDTH - self.other_images['message'].get_width()) / 2, cfg.SCREENHEIGHT * 0.12]
        bird_idx = 0
        bird_idx_change_count = 0

        bird_idx_cycle = itertools.cycle([0, 1, 2, 1])
        bird_pos = [cfg.SCREENWIDTH * 0.2, cfg.SCREENHEIGHT * 0.5]

        bird_y_shift_count = 0

        bird_y_shift_max = 9
        shift = 1

        clock = pygame.time.Clock()
        # if mode == 'train':
        #     return {'bird_pos': bird_pos, 'base_pos': base_pos, 'bird_idx': bird_idx}

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        return {'bird_pos': bird_pos, 'base_pos': base_pos, 'bird_idx': bird_idx}

            bird_idx_change_count += 1
            if bird_idx_change_count % 5 == 0:
                bird_idx = next(bird_idx_cycle)
                bird_idx_change_count = 0

            # 地面的长度比背景图要长，可以利用这个特点在每次画画的时候向左移动一点距离，就能造成视觉上移动很快的错觉
            base_pos[0] = -((-base_pos[0] + 4) % base_diff_bg)

            bird_y_shift_count += 1
            if bird_y_shift_count == bird_y_shift_max:
                bird_y_shift_max = 16
                shift = -1 * shift
                bird_y_shift_count = 0
            bird_pos[-1] = bird_pos[-1] + shift
            self.screen.blit(self.backgroud_image, (0, 0))
            self.screen.blit(list(self.bird_images.values())[bird_idx], bird_pos)
            self.screen.blit(self.other_images['message'], msg_pos)
            self.screen.blit(self.other_images['base'], base_pos)
            pygame.display.update()
            clock.tick(cfg.FPS)


'''parse arguments'''


def parseArgs():
    parser = argparse.ArgumentParser(description='Use q learning to play flappybird')
    parser.add_argument('--mode', dest='mode', help='Choose <train> or <test> please', default='train1', type=str)
    parser.add_argument('--policy', dest='policy', help='Choose <plain> or <greedy> please', default='plain', type=str)
    args = parser.parse_args()
    return args


'''initialize the game'''

'''show the game score'''


def showScore(screen, score, number_images):
    digits = list(str(int(score)))
    width = 0
    for d in digits:
        width += number_images.get(d).get_width()
    offset = (cfg.SCREENWIDTH - width) / 2
    for d in digits:
        screen.blit(number_images.get(d), (offset, cfg.SCREENHEIGHT * 0.1))
        offset += number_images.get(d).get_width()


'''the main function to be called'''


def main(mode, policy, agent, modelpath):
    game_start_info = startGame(screen, sounds, bird_images, other_images, backgroud_image, cfg, mode)

    score = 0
    bird_pos, base_pos, bird_idx = list(game_start_info.values())
    base_diff_bg = other_images['base'].get_width() - backgroud_image.get_width()
    clock = pygame.time.Clock()

    pipe_sprites = pygame.sprite.Group()
    for i in range(2):
        pipe_pos = Pipe.randomPipe(cfg, pipe_images.get('top'))
        pipe_sprites.add(Pipe(image=pipe_images.get('top'),
                              position=(cfg.SCREENWIDTH + 200 + i * cfg.SCREENWIDTH / 2, pipe_pos.get('top')[-1]),
                              type_='top'))
        pipe_sprites.add(Pipe(image=pipe_images.get('bottom'),
                              position=(cfg.SCREENWIDTH + 200 + i * cfg.SCREENWIDTH / 2, pipe_pos.get('bottom')[-1]),
                              type_='bottom'))
    # --the instanced class of bird
    bird = Bird(images=bird_images, idx=bird_idx, position=bird_pos)
    is_add_pipe = True
    is_game_running = True

    while is_game_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if mode == 'train': agent.saveModel(modelpath)
                pygame.quit()
                sys.exit()
        # --use QLearningAgent to play flappybird
        delta_x = 10000
        delta_y = 10000

        for pipe in pipe_sprites:
            if pipe.type_ == 'bottom' and (pipe.rect.left - bird.rect.left + 30) > 0:
                if pipe.rect.right - bird.rect.left < delta_x:
                    delta_x = pipe.rect.left - bird.rect.left
                    delta_y = pipe.rect.top - bird.rect.top

        delta_x = int((delta_x + 60) / 5)
        delta_y = int((delta_y + 225) / 5)
        if agent.act(delta_x, delta_y, int(bird.speed + 9)):
            bird.setFlapped()
            sounds['wing'].play()

        # --check the collide between bird and pipe
        for pipe in pipe_sprites:
            if pygame.sprite.collide_mask(bird, pipe):
                sounds['hit'].play()
                is_game_running = False
        # --update the bird
        boundary_values = [0, base_pos[-1]]
        is_dead = bird.update(boundary_values)
        if is_dead:
            sounds['hit'].play()
            is_game_running = False
        # --if game over, update qvalues_storage
        if not is_game_running:
            agent.update(score, True) if mode == 'train' else agent.update(score, False)
        # --move the bases to the left to achieve the effect of bird flying forward
        base_pos[0] = -((-base_pos[0] + 4) % base_diff_bg)
        # --move the pipes to the left to achieve the effect of bird flying forward
        flag = False
        reward = 1
        for pipe in pipe_sprites:
            pipe.rect.left -= 4
            if pipe.rect.centerx <= bird.rect.centerx and not pipe.used_for_score:
                pipe.used_for_score = True
                score += 0.5
                reward = 5
                if '.5' in str(score):
                    sounds['point'].play()
            if pipe.rect.left < 5 and pipe.rect.left > 0 and is_add_pipe:
                pipe_pos = Pipe.randomPipe(cfg, pipe_images.get('top'))
                pipe_sprites.add(Pipe(image=pipe_images.get('top'), position=pipe_pos.get('top'), type_='top'))
                pipe_sprites.add(Pipe(image=pipe_images.get('bottom'), position=pipe_pos.get('bottom'), type_='bottom'))
                is_add_pipe = False
            elif pipe.rect.right < 0:
                pipe_sprites.remove(pipe)
                flag = True
        if flag: is_add_pipe = True
        # --set reward
        if mode == 'train' and is_game_running: agent.setReward(reward)
        # --blit the necessary game elements on the screen
        screen.blit(backgroud_image, (0, 0))
        pipe_sprites.draw(screen)
        screen.blit(other_images['base'], base_pos)
        showScore(screen, score, number_images)
        bird.draw(screen)
        pygame.display.update()
        clock.tick(cfg.FPS)
    # the end interface of our game
    endGame(screen, sounds, showScore, score, number_images, bird, pipe_sprites, backgroud_image, other_images,
            base_pos, cfg, mode)


'''run'''
if __name__ == '__main__':
    # parse arguments in command line
    args = parseArgs()
    mode = args.mode.lower()
    policy = args.policy.lower()
    # assert mode in ['train', 'test'], '--mode should be <train> or <test>'
    assert policy in ['plain', 'greedy'], '--policy should be <plain> or <greedy>'
    # the instanced class of QLearningAgent or QLearningGreedyAgent, and the path to save and load model
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    agent = QLearningAgent(mode) if policy == 'plain' else QLearningGreedyAgent(mode)
    modelpath = 'checkpoints/qlearning_%s.pkl' % policy
    if os.path.isfile(modelpath):
        agent.loadModel(modelpath)
    # begin game
    while True:
        main(mode, policy, agent, modelpath)
