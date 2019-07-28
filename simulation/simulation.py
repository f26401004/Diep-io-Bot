import sys
sys.path.append('..') 

import pygame
import numpy as np
import random
import util
import math
from DQN import DQNAgent
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from objects import Player
from objects import Diep
from objects import Stuff
from objects import Bullet

display_option = True
speed = 0
pygame.font.init()

class Game:
  def __init__(self, game_width, game_height, agent, record):
    # set the window title
    pygame.display.set_caption('Diep training')
    # set the game field
    self.game_width = game_width
    self.game_height = game_height
    self.field = {
      'width': game_width,
      'height': game_height
    }
    self.game_display = pygame.display.set_mode((game_width, game_height + 60))
    # self.bg = pygame.image.load('images/background.png')

    # generate the player instance
    self.player = Player(400, 300)
    # initialize the map infomation
    self.map_info = {
      'dieps': [],
      'stuffs': [],
      'bullets': [],
      'traps': []
    }

    # initialize the basic infomation of the game
    self.score = 0
    self.record = record
    self.detect_reward = 0
    self.timeout = False

    # initialize the agent action
    self.initialize(agent)

  def initialize(self, agent):
    # get the old state of the agent
    old_state = agent.get_state(self)
    # initialize the action with all zero
    action = [0, 0, 0, 0, 0, 0] # shoot, angle, move up, move down, move left, move right
    # let player do the action
    self.player.do_action(self, action)
    # get the new state after do the action
    new_state = agent.get_state(self)
    # get the reward from the current status
    reward = agent.set_reward(self, self.player.attr['hp'] > 0, self.player.hit)
    # remember the status and replay with the memory
    agent.remember(old_state, action, reward, new_state, self.player.attr['hp'] > 0, self.player.hit)
    agent.replay_new(agent.memory)

    # random generate stuff
    for i in range(50):
      x = random.randint(50, self.game_width - 50)
      y = random.randint(50, self.game_height - 50)
      # check collision
      flag = True
      while flag:
        flag = False
        x = random.randint(50, self.game_width - 50)
        y = random.randint(50, self.game_height - 50)
        position = {
          'x': x,
          'y': y
        }
        if (util.distance(self.player.position, position) < 15 + self.player.radius):
          flag = True
        for stuff in self.map_info['stuffs']:
          if (util.distance(stuff.position, position) < stuff.radius * 2):
            flag = True
      stuff = Stuff(x, y)
      self.map_info['stuffs'].append(stuff)
    
  

  def draw_ui(self):
    # initialize the font instance
    font = pygame.font.SysFont('Segoe UI', 20)
    font_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = font.render('SCORE: ', True, (255, 255, 255))
    text_score_number = font.render(str(self.score), True, (255, 255, 255))
    text_highesst = font.render('HIGHEST SCORE: ', True, (255, 255, 255))
    text_highest_number = font_bold.render(str(self.record), True, (255, 255, 255))
    player_hp = font.render('PLAYER HP: ', True, (255, 255, 255))
    player_hp_number = font.render(str(int(self.player.attr['hp'])), True, (255, 255, 255))
    rest = font.render('REST STUFFS: ', True, (255, 255, 255))
    rest_number = font.render(str(len(self.map_info['stuffs'])), True, (255, 255, 255))
    
    # draw the basic information of the game
    self.game_display.blit(text_score, (45, self.field['height'] + 30))
    self.game_display.blit(text_score_number, (120, self.field['height'] + 30))
    self.game_display.blit(text_highesst, (160, self.field['height']+  30))
    self.game_display.blit(text_highest_number, (300, self.field['height'] + 30))
    self.game_display.blit(player_hp, (400, self.field['height'] + 30))
    self.game_display.blit(player_hp_number, (500, self.field['height'] + 30))
    self.game_display.blit(rest, (550, self.field['height'] + 30))
    self.game_display.blit(rest_number, (670, self.field['height'] + 30))
    # self.game_display.blit(self.bg, (0, 0))

  def display(self):
    # refresh the background color first
    self.game_display.fill((78, 79, 80))
    # draw the ui
    self.draw_ui()
    # draw all element on the map
    self.player.draw(self)
    for diep in self.map_info['dieps']:
      dieps.draw(self)
    for stuff in self.map_info['stuffs']:
      stuff.draw(self)
    for bullet in self.map_info['bullets']:
      bullet.draw(self)

  def update(self):
    # update all object attribute
    self.player.update()
    for diep in self.map_info['dieps']:
      dieps.update()
    for stuff in self.map_info['stuffs']:
      stuff.update()
    for bullet in self.map_info['bullets']:
      bullet.update()
    # detect collisions
    self.deal_with_collisions()
    # gain shoot reward
    for bullet in self.map_info['bullets']:
      if not bullet.owner == self.player.id:
        continue
      if bullet.hit == True:
        self.detect_reward += 100

    # display all object
    self.display()
    pygame.display.update()
    
  def deal_with_collisions(self):
    collisions = []
    player = self.player
    player.hit = False
    # (player, stuff), (player, bullet), (player, diep)
    # (stuff, bullet), (stuff, stuff), (stuff, diep)
    # (bullet, bullet), (bullet, diep)

    
    for stuff in self.map_info['stuffs']:
      stuff.hit = False
      # detect player collide with stuff
      if (util.distance(player.position, stuff.position) < player.radius + stuff.radius):
        player.hit = stuff.hit = True
        if not (((player, stuff) in collisions) or ((stuff, player) in collisions)):
          collisions.append((player, stuff))
      # detect stuff collide with stuff
      for stuff2 in self.map_info['stuffs']:
        if (stuff == stuff2):
          continue
        if (util.distance(stuff.position, stuff2.position) < stuff.radius + stuff2.radius):
          stuff.hit = stuff2.hit = True
          if not (((stuff, stuff2) in collisions) or ((stuff2, stuff) in collisions)):
            collisions.append((stuff, stuff2))
    # detect player collide with bullet
    for bullet in self.map_info['bullets']:
      bullet.hit = False
      if (bullet.owner == player.id):
        continue
      
      if (util.distance(player.position, bullet.position) < player.radius + bullet.radius):
        player.hit = bullet.hit = True
        if not (((player, bullet) in collisions) or ((bullet, player) in collisions)):
          collisions.append((player, bullet))
    
    for bullet in self.map_info['bullets']:
      # detect stuff collide with bullet
      for stuff in self.map_info['stuffs']:
        if (util.distance(bullet.position, stuff.position) < bullet.radius + stuff.radius):
          bullet.hit = stuff.hit = True
          if not (((stuff, bullet) in collisions) or ((bullet, stuff) in collisions)):
            collisions.append((stuff, bullet))
      # detect bullet collide with bullet
      for bullet2 in self.map_info['bullets']:
        if (bullet == bullet2):
          continue
        if (util.distance(bullet.position, bullet2.position) < bullet.radius + bullet2.radius):
          bullet.hit = bullet2.hit = True
          if not (((bullet, bullet2) in collisions) or ((bullet2, bullet) in collisions)):
            collisions.append((bullet, bullet2))
    # deal with all collision pair
    for pair in collisions:
      # physical object collision 
      self.object_collision(pair)
      # call the collide_with function
      pair[0].collide_with(pair[1])
      pair[1].collide_with(pair[0])

  def object_collision(self, pair):
    copy = pair[0].velocity.copy()
    print('pair[0].velocity: ', copy)
    print('pair[1].velocity: ', pair[1].velocity)
    pair[0].velocity['x'] = pair[1].velocity['x']
    pair[0].velocity['y'] = pair[1].velocity['y']
   
    pair[1].velocity['x'] = copy['x']
    pair[1].velocity['y'] = copy['y']

    copy = pair[0].acceleration.copy()
    pair[0].acceleration['up'] = pair[1].acceleration['up']
    pair[0].acceleration['down'] = pair[1].acceleration['down']
    pair[0].acceleration['left'] = pair[1].acceleration['left']
    pair[0].acceleration['right'] = pair[1].acceleration['right']

    pair[1].acceleration['up'] = copy['up']
    pair[1].acceleration['down'] = copy['down']
    pair[1].acceleration['left'] = copy['left']
    pair[1].acceleration['right'] = copy['right']



  def plot_seaborn(self, array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()

def start():
  pygame.init()
  agent = DQNAgent()
  counter_games = 0
  score_plot = []
  counter_plot = []
  record = 0
  while counter_games < 150:
    game = Game(800, 600, agent, record)

    if display_option:
      game.display()
    def timeout ():
      game.timeout = True
    timer = threading.Timer(150, timeout)
    timer.start()
    while not (game.player.attr['hp'] <= 0 or len(game.map_info['stuffs']) == 0 or game.timeout):
      agent.epslion = 80 - counter_games

      old_state = agent.get_state(game)

      if random.randint(0, 200) < agent.epslion:
        final_action = to_categorical(random.randint(0, 5), num_classes=6)
        # random shoot angle
        final_action[1] = random.uniform(0, 1)
      else:
        prediction = agent.model.predict(old_state.reshape((1, 6 + agent.grid_size * agent.grid_size)))
        final_action = to_categorical(np.argmax(prediction[0]), num_classes=6)
        # use original predict value as shoot angle
        final_action[1] = prediction[0][1]
      
      game.player.do_action(game, final_action)
      new_state = agent.get_state(game)
      reward = agent.set_reward(game, game.player.attr['hp'] <= 0, game.player.hit) + game.detect_reward
      game.detect_reward = 0
      print('current reward: ', reward)
      agent.train_short_memory(old_state, final_action, reward, new_state, game.player.attr['hp'] <= 0, game.player.hit)
      agent.remember(old_state, final_action, reward, new_state, game.player.attr['hp'] <= 0, game.player.hit)
      record = max(game.score, record)
      if display_option:
        game.update()
        pygame.time.wait(speed)
    
    agent.replay_new(agent.memory)
    counter_games += 1
    print('Game %d, Score: %d' % (counter_games, game.score))
    score_plot.append(game.score)
    counter_plot.append(counter_games)

  agent.model.save_weights('weights.hdf5')
  game.plot_seaborn(counter_plot, score_plot)


start()