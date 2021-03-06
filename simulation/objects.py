import random
import json
import uuid
import math
import pygame
import numpy as np
import util

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

class PlayerShoot(GameObject):
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
    self.shoot_status = {
      'fire': False,
      'angle': 0,
      'cd': 0
    }

  def update(self):
    self.shoot_status['cd'] -= 0 if self.shoot_status['cd'] <= 0 else 1

  def draw(self, game):
    game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))

  def do_shoot_action(self, game, action):
    self.shoot_status['fire'] = False
    if action[0] == 1 and self.shoot_status['cd'] <= 0:
      self.shoot_status['fire'] = True
      self.shoot_status['angle'] = action[1]
      existence = (self.status['bullet_penetration'] - 1) * 5 + 20
      bullet = Bullet(self.position['x'], self.position['y'], existence, self.status['bullet_damage'], {
        'x': math.cos(action[1]) * (10 + self.status['bullet_speed']),
        'y': math.sin(action[1]) * (10 + self.status['bullet_speed'])
      }, self.id)
      game.map_info['bullets'].append(bullet)
      self.shoot_status['cd'] = 50 * math.log(self.status['bullet_reload'] + 1, 10)
  
  def collide_with(self, collider):
    self.attr['hp'] -= collider.attr['body_damage'] * 5


class PlayerMove(GameObject):
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
    self.shoot_status = {
      'fire': False,
      'angle': 0,
      'cd': 0
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

    if self.acceleration['down'] - self.acceleration['up'] > 0:
      self.velocity['y'] = min((self.acceleration['down'] - self.acceleration['up']) * friction, math.sqrt(self.status['move_speed'] + 5))
    else:
      self.velocity['y'] = max((self.acceleration['down'] - self.acceleration['up']) * friction, math.sqrt(self.status['move_speed'] + 5) * (-1))
    if self.acceleration['right'] - self.acceleration['left'] > 0:
      self.velocity['x'] = min((self.acceleration['right'] - self.acceleration['left']) * friction, math.sqrt(self.status['move_speed'] + 5))
    else:
      self.velocity['x'] = min((self.acceleration['right'] - self.acceleration['left']) * friction, math.sqrt(self.status['move_speed'] + 5)* (-1))

    #   self.velocity['x'] = min((self.acceleration['right'] - self.acceleration['left']) * friction, math.sqrt(self.status['move_speed'] + 5))
    # else:
    #   self.velocity['x'] = min((self.acceleration['right'] - self.acceleration['left']) * friction, math.sqrt(self.status['move_speed'] + 5) * (-1))

    
    self.position['x'] = min(max(self.position['x'] + self.velocity['x'], 0), 800)
    self.position['y'] = min(max(self.position['y'] + self.velocity['y'], 0), 600)
    self.shoot_status['cd'] -= 0 if self.shoot_status['cd'] <= 0 else 1
    self.attr['hp'] += math.log(self.status['hp_regeneration'] + 1) if self.attr['hp'] <= self.attr['maxhp'] else 0
    self.attr['hp'] = max(min(self.attr['hp'], self.attr['maxhp']), 0)

  def draw(self, game):
    if (self.position['x'] == 0 or self.position['x'] == game.game_width or self.position['y'] == 0 or self.position['y'] == game.game_height) and game.agent.reward < 0.43:
      self.attr['hp'] = 0
    # if self.attr['hp'] > 0:
    game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))

  def do_move_action(self, game, action):
    # if action[0] == 1 and self.shoot_status['cd'] <= 0:
    #   self.shoot_status['fire'] = False
    #   self.shoot_status['angle'] = action[1]
    #   existence = (self.status['bullet_penetration'] - 1) * 5 + 20
    #   bullet = Bullet(self.position['x'], self.position['y'], existence, self.status['bullet_damage'], {
    #     'x': math.cos(action[1]) * (10 + self.status['bullet_speed']),
    #     'y': math.sin(action[1]) * (10 + self.status['bullet_speed'])
    #   }, self.id)
    #   game.map_info['bullets'].append(bullet)
    #   self.shoot_status['cd'] = 50 * math.log(self.status['bullet_reload'] + 1, 10)
    # else:
    #   self.shoot_status['fire'] = False
    
    self.move_direction = {
      'up': False,
      'down': False,
      'left': False,
      'right': False
    }
    type_num = np.argwhere(action == 1)
    if type_num == 1:
      self.move_direction['up'] = True
    elif type_num == 2:
      self.move_direction['down'] = True
    elif type_num == 3:
      self.move_direction['left'] = True
    elif type_num == 4:
      self.move_direction['right'] = True
    elif type_num == 5:
      self.move_direction['up'] = True
      self.move_direction['left'] = True
    elif type_num == 6:
      self.move_direction['up'] = True
      self.move_direction['right'] = True
    elif type_num == 7:
      self.move_direction['down'] = True
      self.move_direction['left'] = True
    elif type_num == 8:
      self.move_direction['down'] = True
      self.move_direction['right'] = True
    
    target = None
    angle = -1
    # find the closest stuff and shoot
    for stuff in game.map_info['stuffs']:
      dist = util.distance(self.position, stuff.position)
      if target:
        if dist < 200 and util.distance(self.position, target.position) > dist:
          angle = util.angle(self.position, stuff.position)
          target = stuff
      else:
        if dist < 200:
          angle = util.angle(self.position, stuff.position)
          target = stuff
    if target:
      self.do_shoot_action(game, [1, angle])


  def do_shoot_action(self, game, action):
    self.shoot_status['fire'] = False
    if action[0] == 1 and self.shoot_status['cd'] <= 0:
      self.shoot_status['fire'] = True
      self.shoot_status['angle'] = action[1]
      existence = (self.status['bullet_penetration'] - 1) * 5 + 20
      bullet = Bullet(self.position['x'], self.position['y'], existence, self.status['bullet_damage'], {
        'x': math.cos(action[1]) * (10 + self.status['bullet_speed']),
        'y': math.sin(action[1]) * (10 + self.status['bullet_speed'])
      }, self.id)
      game.map_info['bullets'].append(bullet)
      self.shoot_status['cd'] = 50 * math.log(self.status['bullet_reload'] + 1, 10)

  def collide_with(self, collider):
    self.attr['hp'] -= collider.attr['body_damage'] * 5

class Diep(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
    self.image = pygame.image.load('images/diep.png')
  
class Stuff(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
    self.radius = 15
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

    self.velocity['x'] = (self.velocity['x']) * friction
    self.velocity['y'] = (self.velocity['y']) * friction

    self.position['x'] = min(max(self.position['x'] + self.velocity['x'], 0), 800)
    self.position['y'] = min(max(self.position['y'] + self.velocity['y'], 0), 600)

  def draw(self, game):
    if self.attr['hp'] > 0:
      game.game_display.blit(self.image, (self.position['x'] - self.radius, self.position['y'] - self.radius))
    else:
      game.score += 1
      game.map_info['stuffs'].remove(self)

  def collide_with(self, collider):
    if (type(collider) == PlayerMove or type(collider) == PlayerShoot):
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
