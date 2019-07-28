import random
import json
import uuid
import math
import pygame

friction = 0.97

class GameObject(object):
  def __init__(self, x, y):
    self.id = uuid.uuid4()
    self.radius = 15
    self.hit = False
    self.image = pygame.image.load('images/object.png')
    self.velocity = {
      'x': 0.0,
      'y': 0.0
    }
    self.position = {
      'x': x,
      'y': y
    }
    self.acceleration = {
      'up': 0.0,
      'down': 0.0,
      'left': 0.0,
      'right': 0.0
    }
    self.move_direction = {
      'up': False,
      'down': False,
      'left': False,
      'right': False
    }

  def move(self, type_str):
    self.move_direction[type_str] = True
  
  def update(self):
    if (self.move_direction['up']):
      self.acceleration['up'] = 10
    if (self.move_direction['down']):
      self.acceleration['down'] = 10
    if (self.move_direction['left']):
      self.acceleration['left'] = 10
    if (self.move_direction['right']):
      self.acceleration['right'] = 10

    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction

    self.position['x'] = min(max(self.position['x'] + self.velocity['x'], 0), 800)
    self.position['y'] = min(max(self.position['y'] + self.velocity['y'], 0), 600)

  def draw(self, game):
    pass

  def collide_with(self, collider):
    pass

class Player(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
    self.radius = 15
    self.image = pygame.image.load('images/player.png')
    self.attr = {
      'hp': 100,
      'maxhp': 100,
      'level': 1,
      'exp': 0,
      'shoot_cd': 0
    }
    self.status = {
      'maxhp': 1,
      'hp_regeneration': 1,
      'move_speed': 1,
      'bullet_speed': 1,
      'bullet_penetration': 1,
      'bullet_reload': 1,
      'bullet_damage': 1,
      'body_damage': 1
    }

  def update(self):
    if (self.move_direction['up']):
      self.acceleration['up'] = 10
    if (self.move_direction['down']):
      self.acceleration['down'] = 10
    if (self.move_direction['left']):
      self.acceleration['left'] = 10
    if (self.move_direction['right']):
      self.acceleration['right'] = 10
    self.acceleration['up'] = self.acceleration['up'] * friction
    self.acceleration['down'] = self.acceleration['down'] * friction
    self.acceleration['left'] = self.acceleration['left'] * friction
    self.acceleration['right'] = self.acceleration['right'] * friction

    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction
    self.position['x'] = min(max(self.position['x'] + self.velocity['x'], 0), 800)
    self.position['y'] = min(max(self.position['y'] + self.velocity['y'], 0), 600)
    self.attr['shoot_cd'] -= 0 if self.attr['shoot_cd'] <= 0 else 1
    self.attr['hp'] += math.log(self.status['hp_regeneration'] + 1) if self.attr['hp'] <= self.attr['maxhp'] else 0
    self.attr['hp'] = min(self.attr['hp'], self.attr['maxhp'])

  def draw(self, game):
    if self.attr['hp'] > 0:
      game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))

  def do_action(self, game, action):
    
    if action[0] == 1 and self.attr['shoot_cd'] <= 0:
      existence = (self.status['bullet_penetration'] - 1) * 5 + 20
      bullet = Bullet(self.position['x'], self.position['y'], existence, self.status['bullet_damage'], {
        'x': math.cos(2 * math.pi * action[1]) * (10 + self.status['bullet_speed']),
        'y': math.sin(2 * math.pi * action[1]) * (10 + self.status['bullet_speed'])
      }, self.id)
      game.map_info['bullets'].append(bullet)
      self.attr['shoot_cd'] = 50 * math.log(self.status['bullet_reload'] + 1, 10)
    
    self.move_direction = {
      'up': False,
      'down': False,
      'left': False,
      'right': False
    }
    if action[2] == 1:
      self.move_direction['up'] = True
    if action[3] == 1:
      self.move_direction['down'] = True
    if action[4] == 1:
      self.move_direction['left'] = True
    if action[5] == 1:
      self.move_direction['right'] = True
  
  def collide_with(self, collider):
    self.attr['hp'] -= collider.attr['body_damage'] * 5

class Diep(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
    self.image = pygame.image.load('images/diep.png')
  
class Stuff(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
    self.rafius = 15
    self.image = pygame.image.load('images/stuff.png')

    reader = open('stuffType.json', 'r')
    stuff_type = json.loads(reader.read())
    type_number = 1#random.randint(1, 5)
    self.attr = {
      'hp': stuff_type[str(type_number)]['HP'],
      'exp': stuff_type[str(type_number)]['EXP'],
      'body_damage': stuff_type[str(type_number)]['BodyDamage']
    }
  
  def update(self):
    self.acceleration['up'] = self.acceleration['up'] * friction
    self.acceleration['down'] = self.acceleration['down'] * friction
    self.acceleration['left'] = self.acceleration['left'] * friction
    self.acceleration['right'] = self.acceleration['right'] * friction

    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction

    self.position['x'] = min(max(self.position['x'] + self.velocity['x'], 0), 800)
    self.position['y'] = min(max(self.position['y'] + self.velocity['y'], 0), 600)

  def draw(self, game):
    if self.attr['hp'] > 0:
      game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))
    else:
      game.score += 1
      game.detect_reward += 500
      game.map_info['stuffs'].remove(self)

  def collide_with(self, collider):
    if (type(collider) == Player):
      self.attr['hp'] -= math.log(collider.status['body_damage'] + 1)
    elif (type(collider) == Bullet):
      self.attr['hp'] -= collider.attr['body_damage'] * 5
    else:
      self.attr['hp'] -= collider.attr['body_damage'] * 5
      

class Bullet(GameObject):
  def __init__(self, x, y, hp, damage, velocity, owner):
    GameObject.__init__(self, x, y)
    self.radius = 5
    self.owner = owner
    self.image = pygame.image.load('images/bullet.png')
    self.velocity = velocity
    self.acceleration = {
      'up': math.log(velocity['y'] * -1) if velocity['y'] < 0 else 0,
      'down': math.log(velocity['y']) if velocity['y'] > 0 else 0,
      'left': math.log(velocity['x'] * -1) if velocity['x'] < 0 else 0,
      'right': math.log(velocity['x']) if velocity['x'] > 0 else 0,
    }

    self.attr = {
      'hp': hp,
      'body_damage': damage
    }
  
  def update(self):
    self.attr['hp'] -= 1

    self.position['x'] = self.position['x'] + self.velocity['x']
    self.position['y'] = self.position['y'] + self.velocity['y']

  def draw(self, game):
    if self.attr['hp'] > 0:
      game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))
    else:
      game.map_info['bullets'].remove(self)
  
  def collide_with(self, collider):
    self.attr['hp'] = 0
