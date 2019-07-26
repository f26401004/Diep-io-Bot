import pygame
import numpy as np
import random
from ...DQN import DQNAgent
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from objects import player

display_option = False
speed = 0
pygame.font.init()


class Game:
  def __init__(self, game_width, game_height):
    pygame.display.set_caption('Diep training')
    self.game_width = game_width
    self.game_height = game_weight
    self.game_display = pygame.display.set_mode((game_weight, game_height + 60))
    self.bg = pygame.image.load('image/background.png')
    self.crash = False
    self.player = player()
    # TODO: random generate stuff and traps
    self.score = 0

  