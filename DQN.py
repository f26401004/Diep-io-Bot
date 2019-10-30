from keras.optimizers import Adam
# from keras_radam import RAdam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
import random
import numpy as np
import pandas as pd
import os
import math
import util
from operator import add
# import keras
import pygame
 
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 8} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 
 
class DQNMoveAgent(object):
  def __init__(self):
    self.epslion = 0
    self.reward = 0
    self.gamma = 0.9
    self.grid_size = 25
    self.dataframe = pd.DataFrame()
    self.short_memory = np.array([])
 
    self.avoid = False
    self.avoid_target = None
    self.old_target = None
    self.agency_target = 1
    self.agency_predict = 0
    self.shoot = False
    self.old_shoot = False
 
    self.learning_rate = 0.001
    weights = None
    if (os.path.isfile('./move_weights.hdf5')):
      weights = 'move_weights.hdf5'
    self.model = self.network(weights)
    self.eplison = 0
    self.actual = []
    self.memory = []
    self.prev_pos = {
      'x': 0,
      'y': 0
    }
    self.stopping = False
    self.stopping_step = 0
 
  def get_state(self, game):
    self.stopping = True
    # record previous position
    if not round(game.player.position['x'], 2) == round(self.prev_pos['x'], 2):
      self.prev_pos['x'] = game.player.position['x']
      self.stopping = False
      self.stopping_step = 0
    if not round(game.player.position['y'], 2) == round(self.prev_pos['y'], 2):
      self.prev_pos['y'] = game.player.position['y']
      self.stopping = False
      self.stopping_step = 0

    if self.stopping:
      self.stopping_step += 1

    # screen = pygame.surfarray.array2d(game.game_display)
    # return screen.reshape(screen.shape[0], screen.shape[1], 1)
    
    target_pos = util.dense_position(game, 5)
    state = [
      # game.player.shoot_status['fire'], # player shoot fire
      # game.player.shoot_status['angle'], # player shoot angle
      # game.player.shoot_status['cd'], # player shoot CD time
      # game.player.attr['hp'] / game.player.attr['maxhp'], # player hp
      round(game.player.position['x'] / game.game_width, 2), # x position
      round(game.player.position['y'] / game.game_height, 2), # y position
      round(game.player.velocity['x'] / math.sqrt(game.player.status['move_speed'] + 5), 2), # x movement
      round(game.player.velocity['y'] / math.sqrt(game.player.status['move_speed'] + 5), 2), # y movement
      round(game.player.acceleration['up'] / 10, 2), # up acceleration volumn
      round(game.player.acceleration['down'] / 10, 2), # down acceleration volumn
      round(game.player.acceleration['left'] / 10, 2), # left acceleration volumn
      round(game.player.acceleration['right'] / 10, 2), # right acceleration volumn
      round(abs(game.player.position['x'] - (game.game_width / 2)) / (game.game_width / 2), 2), # player x distance ratio to center
      round(abs(game.player.position['y'] - (game.game_width / 2)) / (game.game_height / 2), 2), # player y distance ratio to center
      round(abs(game.player.position['x'] - target_pos['x']) / game.game_width, 2), # target position x
      round(abs(game.player.position['y'] - target_pos['y']) / game.game_height, 2), # target position y
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
    # compute the grid-cell value
    v_x = 0 #max(game.player.position['x'] - game.field['width'] / 2, 0)
    v_y = 0 #max(game.player.position['y'] - game.field['height'] / 2, 0)
    player_x = game.player.position['x']
    player_y = game.player.position['y']
    
    weight = 0
    for stuff in game.map_info['stuffs']:
      if not util.distance(game.player.position, stuff.position):
        continue
      weight += 1 / util.distance(game.player.position, stuff.position)
      angle = util.angle(game.player.position, stuff.position)
      angle += 2 * math.pi if angle < 0 else 0
      if 0 < angle <= math.pi / 4:
        state[12] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 4 < angle <= math.pi / 2:
        state[13] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 2 < angle <= math.pi / 4 * 3:
        state[14] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 4 * 3 < angle <= math.pi:
        state[15] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi < angle <= math.pi / 4 * 5:
        state[16] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 4 * 5 < angle <= math.pi / 2 * 3:
        state[17] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 2 * 3 < angle <= math.pi / 4 * 7:
        state[18] += 1 / util.distance(game.player.position, stuff.position)
      if math.pi / 4 * 7 < angle <= math.pi * 2:
        state[19] += 1 / util.distance(game.player.position, stuff.position)
    if weight > 0:
      state[12] = round(state[12] / weight, 2)
      state[13] = round(state[13] / weight, 2)
      state[14] = round(state[14] / weight, 2)
      state[15] = round(state[15] / weight, 2)
      state[16] = round(state[16] / weight, 2)
      state[17] = round(state[17] / weight, 2)
      state[18] = round(state[18] / weight, 2)
      state[19] = round(state[19] / weight, 2)
      


      # for diep in game.map_info['dieps']:
      #   if (current_x < diep.position['x'] < current_x + game.field['width'] / self.grid_size and
      #       current_y < diep.position['y'] < current_y + game.field['height'] / self.grid_size):
      #     if (diep.id == game.player.id):
      #       continue
      #     score += 3
      # for stuff in game.map_info['stuffs']:
      #   if (current_x < stuff.position['x'] < current_x + game.field['width'] / self.grid_size and
      #       current_y < stuff.position['y'] < current_y + game.field['height'] / self.grid_size):
      #     score += 1
      # for bullet in game.map_info['bullets']:
      #   if (current_x < bullet.position['x'] < current_x + game.field['width'] / self.grid_size and
      #     current_y < bullet.position['y'] < current_y + game.field['height'] / self.grid_size and
      #     not bullet.owner == game.player.id):
      #     score -= 5
      # for trap in game.map_info['traps']:
      #   if (current_x < trap.position['x'] < current_x + game.field['width'] / self.grid_size and
      #     current_y < trap.position['y'] < current_y + game.field['height'] / self.grid_size):
      #     score += 1
     
      # state.append(score / 50)
 
    return np.asarray(state)
 
  def set_reward(self, game, dead, damage):
    self.reward = 0.001
    if dead:
      self.reward -= 10
    if damage:
      self.reward -= 10
    target_pos = util.dense_position(game, 5)
    # compute the distance between player and the target position
    # dist = util.distance(game.player.position, target_pos)
    # if dist < 200:
    #   self.reward += 10 / math.log(dist, 10) if dist > 10 else 1
    for stuff in game.map_info['stuffs']:
      dist = util.distance(game.player.position, stuff.position)
      if dist < game.player.radius * 2.5:
        self.reward += -1 / math.log(dist, 10) if dist > 10 else 1

    if self.stopping_step > 0 and self.reward < 0.4:
      self.reward -= self.stopping_step / 1000
    # check the stuff by direction, and set the give the reward
    # temp = 0
    # if game.player.move_direction['up']:
    #   for i in range(max(player_grid_x - 5, 0), min(player_grid_x + 5, self.grid_size)):
    #     for j in range(0, player_grid_y):
    #       temp += current_map[i + j * self.grid_size]
    #   self.reward += temp
    # if game.player.move_direction['down']:
    #   for i in range(max(player_grid_x - 5, 0), min(player_grid_x + 5, self.grid_size)):
    #     for j in range(player_grid_y, self.grid_size):
    #       temp += current_map[i + j * self.grid_size]
    #   self.reward += temp
    # if game.player.move_direction['left']:
    #   for i in range(0, player_grid_x):
    #     for j in range(max(player_grid_y - 5, 0), min(player_grid_y + 5, self.grid_size)):
    #       temp += current_map[i + j * self.grid_size]
    #   self.reward += temp
    # if game.player.move_direction['right']:
    #   for i in range(player_grid_x, self.grid_size):
    #     for j in range(max(player_grid_y - 5, 0), min(player_grid_y + 5, self.grid_size)):
    #       temp += current_map[i + j * self.grid_size]
    #   self.reward += temp
    # set the shoot reward
    return self.reward
 
 
  def network(self, weights=None):
    model = Sequential()
    # model.add(Conv2D(32, 3, 1, input_shape=(800, 660, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(Conv2D(32, 3, 1, activation='relu'))
    # model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(Conv2D(64, 3, 1, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(output_dim=128, activation = 'relu'))
    # model.add(Dense(output_dim=9, activation = 'softmax'))
    model.add(Dense(output_dim=16, activation='relu', input_dim=20))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=32, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=32, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=16, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=9, activation='softmax'))
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)
 
    if weights:
      model.load_weights(weights)
    return model
  def remember(self, state, action, reward, next_state, done, hit):
    self.memory.append((state, action, reward, next_state, done, hit))
 
  def replay_new(self, memory, number):
    if len(memory) > number:
      minibatch = random.sample(memory, number)
    else:
      minibatch = memory
  
    for state, action, reward, next_state, done, hit in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
      target_f = self.model.predict(np.array([state]))
      target_f[0][np.argmax(action)] = target
      self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
 
  def train_short_memory(self, state, action, reward, next_state, done, hit):
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 20)))[0])
    target_f = self.model.predict(state.reshape((1, 20)))
    target_f[0][np.argmax(action)] = target
    self.model.fit(state.reshape((1, 20)), target_f, epochs=1, verbose=0)
  



class DQNShootAgent(object):
  def __init__(self):
    self.epslion = 0
    self.reward = 0
    self.gamma = 0.9
    self.grid_size = 25
    self.dataframe = pd.DataFrame()
    self.short_memory = np.array([])
 
    self.avoid = False
    self.avoid_target = None
    self.old_target = None
    self.agency_target = 1
    self.agency_predict = 0
    self.shoot = False
    self.old_shoot = False
 
    self.learning_rate = 0.005
    weights = None
    if (os.path.isfile('./shoot_weights.hdf5')):
      weights = 'shoot_weights.hdf5'
    self.model = self.network(weights)
    self.eplison = 0
    self.actual = []
    self.memory = []
    self.prev_pos = {
      'x': 0,
      'y': 0
    }
    self.stopping = False
    self.stopping_step = 0
 
  def get_state(self, game):
    self.stopping = True
    if game.player.shoot_status['fire']:
      self.stopping = False
      self.stopping_step = 0
    if self.stopping:
      self.stopping_step += 1
    state = [
      game.map_info['stuffs'][0].position['x'] / game.game_width,
      game.map_info['stuffs'][0].position['y'] / game.game_height,
      game.player.position['x'] / game.game_width,
      game.player.position['y'] / game.game_height,
      util.angle(game.player.position, game.map_info['stuffs'][0].position) / (math.pi * 2)
    ]
    return np.asarray(state)
 
  def set_reward(self, game, dead, damage):
    self.reward = 0
    if game.player.shoot_status['fire']:
      scores = []
      for stuff in game.map_info['stuffs']:
        if util.distance(game.player.position, stuff.position) < 200:
          scores.append(util.angle_score(game.player.shoot_status['angle'], game.player.position, stuff.position))
      print("scores:", scores)
      self.reward = max(scores) * 100 if max(scores) > 0 else max(scores)
    if self.stopping_step > 0:
      self.reward -= self.stopping_step / 1000
    return self.reward
 
 
  def network(self, weights=None):
    model = Sequential()
    model.add(Dense(output_dim=16, activation='relu', input_dim=5))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=32, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=16, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=8, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=1))
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)
 
    if weights:
      model.load_weights(weights)
    return model
  def remember(self, state, action, reward, next_state, done, hit):
    self.memory.append((state, action, reward, next_state, done, hit))
 
  def replay_new(self, memory):
    if len(memory) > 1000:
      minibatch = random.sample(memory, 1000)
    else:
      minibatch = memory
  
    for state, action, reward, next_state, done, hit in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
      target_f = self.model.predict(np.array([state]))
      target_f[0][0] = target
      self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
 
  def train_short_memory(self, state, action, reward, next_state, done, hit):
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 5)))[0])
    target_f = self.model.predict(state.reshape((1, 5)))
    target_f[0][0] = target
    self.model.fit(state.reshape((1, 5)), target_f, epochs=1, verbose=0)
  

