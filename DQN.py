from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
import os
import math
import util
from operator import add

class DQNAgent(object):
  def __init__(self):
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

    self.learning_rate = 0.0005
    weights = None
    if (os.path.isfile('./weights.hdf5')):
      weights = 'weights.hdf5'
    self.model = self.network(weights)
    self.eplison = 0
    self.actual = []
    self.memory = []

  def get_state(self, game):
    state = [
      game.player.attr['shoot_cd'], # player shoot CD time
      game.player.attr['hp'], # player hp
      game.player.position['x'], # x position
      game.player.position['y'], # y position
      game.player.velocity['x'], # x movement
      game.player.velocity['y'], # y movement
    ]
    # compute the grid-cell value
    v_x = max(game.player.position['y'] - game.field['width'] / 2, 0)
    x_y = max(game.player.position['y'] - game.field['height'] / 2, 0)
    for i in range(self.grid_size * self.grid_size):
      score = 0
      current_x = v_x + (i % self.grid_size) * game.field['width'] / self.grid_size
      current_y = v_x + (i / self.grid_size) * game.field['height'] / self.grid_size
      for diep in game.map_info['dieps']:
        if (current_x < diep.position['x'] < current_x + game.field['width'] / self.grid_size and
          current_y < diep.position['y'] < current_y + game.field['height'] / self.grid_size):
          if (diep.id == game.player.id):
            continue
          score += 3
      for stuff in game.map_info['stuffs']:
        if (current_x < stuff.position['x'] < current_x + game.field['width'] / self.grid_size and
          current_y < stuff.position['y'] < current_y + game.field['height'] / self.grid_size):
          score += 2
      for bullet in game.map_info['bullets']:
        if (current_x < bullet.position['x'] < current_x + game.field['width'] / self.grid_size and
          current_y < bullet.position['y'] < current_y + game.field['height'] / self.grid_size):
          score -= 5
      for trap in game.map_info['traps']:
        if (current_x < trap.position['x'] < current_x + game.field['width'] / self.grid_size and
          current_y < trap.position['y'] < current_y + game.field['height'] / self.grid_size):
          score += 1
      
      state.append(score)


    return np.asarray(state)

  def set_reward(self, game, dead, damage):
    current_map = self.get_state(game)[6:]
    player_grid_x = int(game.player.position['x'] / (800 / self.grid_size))
    player_grid_y = int(game.player.position['y'] / (600 / self.grid_size))


    self.reward = 0
    if dead:
      self.reward -= 500
    if damage:
      self.reward -= 20
    # check the stuff by direction, and set the give the reward
    temp = 0
    if game.player.move_direction['up']:
      for i in range(0, self.grid_size):
        for j in range(0, player_grid_y):
          distance = math.sqrt((i * (800 / self.grid_size))**2 + (j * (600 / self.grid_size))**2)
          if distance < (800 / self.grid_size) * 5:
            temp += current_map[i + j * self.grid_size]
      self.reward += temp * 5
    if game.player.move_direction['down']:
      for i in range(0, self.grid_size):
        for j in range(player_grid_y, self.grid_size):
          distance = math.sqrt((i * (800 / self.grid_size))**2 + (j * (600 / self.grid_size))**2)
          if distance < (800 / self.grid_size) * 5:
            temp += current_map[i + j * self.grid_size]
      self.reward += temp * 5
    if game.player.move_direction['left']:
      for i in range(0, player_grid_x):
        for j in range(0, self.grid_size):
          distance = math.sqrt((i * (800 / self.grid_size))**2 + (j * (600 / self.grid_size))**2)
          if distance < (800 / self.grid_size) * 5:
            temp += current_map[i + j * self.grid_size]
      self.reward += temp * 5
    if game.player.move_direction['right']:
      for i in range(player_grid_x, self.grid_size):
        for j in range(0, self.grid_size):
          distance = math.sqrt((i * (800 / self.grid_size))**2 + (j * (600 / self.grid_size))**2)
          if distance < (800 / self.grid_size) * 5:
            temp += current_map[i + j * self.grid_size]
      self.reward += temp * 5
    
    return self.reward


  def network(self, weights=None):
    model = Sequential()  
    model.add(Dense(output_dim=200, activation='relu', input_dim=6+self.grid_size**2))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=150, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=100, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=50, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(output_dim=6, activation='softmax'))
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
      target_f[0][np.argmax(action)] = target
      self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

  def train_short_memory(self, state, action, reward, next_state, done, hit):
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 6 + self.grid_size**2)))[0])
    target_f = self.model.predict(state.reshape((1, 6 + self.grid_size**2)))
    target_f[0][np.argmax(action)] = target
    self.model.fit(state.reshape((1, 6 + self.grid_size**2)), target_f, epochs=1, verbose=0)
    