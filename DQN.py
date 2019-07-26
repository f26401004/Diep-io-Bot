from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Droput
import random
import numpy as np
import pandas as pd
from opertor import add

class DQNAgent(object):
  def __init__(self):
    self.reward = 0
    self.gamma = 0.9
    self.grid_size = 5
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
    self.model = self.network()
    self.eplison = 0
    self.actual = []
    self.memory = []

  def get_state(self, game, player):
    state = [
      player.shootCD, # player shoot CD time
      player.status.bulletDamage, # player bullet damage level
      player.gameObject.position.x, # x position
      player.gameObject.position.y, # y position
      player.gameObject.velocity.x, # x movement
      player.gameObject.velocity.x, # y movement
    ]
    # compute the grid-cell value
    v_x = max(player.gameObject.position.y - game.field.width / 2, 0)
    x_y = max(player.gameObject.position.y - game.field.height / 2, 0)
    for i in range(self.grid_size * self.grid_size):
      score = 0
      current_x = v_x + (i % self.grid_size) * game.field.width / self.grid_size
      current_y = v_x + (i / self.grid_size) * game.field.height / self.grid_size
      for diep in game.dieps:
        if (current_x < diep.gameObject.position.x < current_x + game.field.width / self.grid_size and
          current_y < diep.gameObject.position.y < current_y + game.field.height / self.grid_size):
          if (diep.gameObject.id == player.gameObject.id):
            continue
          score += 3
      for stuff in game.stuffs:
        if (current_x < stuff.gameObject.position.x < current_x + game.field.width / self.grid_size and
          current_y < stuff.gameObject.position.y < current_y + game.field.height / self.grid_size):
          score += 2
      for bullet in game.bullets:
        if (current_x < bullet.gameObject.position.x < current_x + game.field.width / self.grid_size and
          current_y < bullet.gameObject.position.y < current_y + game.field.height / self.grid_size):
          score -= 5
      for trap in game.traps:
        if (current_x < trap.gameObject.position.x < current_x + game.field.width / self.grid_size and
          current_y < trap.gameObject.position.y < current_y + game.field.height / self.grid_size):
          score += 1
      
      state.append(score)


    return np.asarray(state)

  def set_reward(self, player, dead, damage):
    self.reward = 0
    if dead:
      self.reward -= 10
      return self.reward
    elif damage:
      self.reward -= 3
    else:
      self.reward = 10
    return self.reward


  def network(self, weights=None):
    model = Sequential()  
    model.add(Dense(output_dim=120, activation='relu', input_dim=11))
    model.add(Droput(0.15))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Droput(0.15))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Droput(0.15))
    model.add(Dense(output_dim=3, activation='softmax'))
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)

    if weights:
      model.load_weights(weights)
    return model
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def replay_new(self, memory):
    if len(memory) > 1000:
      minibatch = random.sample(memory, 1000)
    else:
      minibatch = memory
    
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
      target_f = self.model.predict(np.array[state])
      target_f[0][np.argmax(action)] = target
      self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

  def train_short_memory(self, state, action, reward, next_state, done):
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 6 + self.grid_size * self.grid_size)))[0])
    target_f = self.model.predict(state.reshape((1, 6 + self.grid_size * self.grid_size)))
    target_f[0][np.argmax(action)] = target
    self.model.fit(state.reshape((1, 6 + self.grid_size * self.grid_size)), target_f, epochs=1, verbose=0)
    