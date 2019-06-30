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
    self.gamma = 0.0
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
      player.gameObject.velocity.x < 0,
      player.gameObject.velocity.x > 0,
      player.gameObject.velocity.y < 0,
      player.gameObject.velocity.y > 0,
      
    ]

  def set_reward(self, player, dead):
    self.reward = 0
    if dead:
      self.reward -= 10
      return self.reward
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