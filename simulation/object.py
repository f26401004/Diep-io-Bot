import random
import json

friction = 0.97

class GameObject(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.velocity = {
      x: 0.0,
      y: 0.0
    }
    self.acceleration = {
      'up': 0.0,
      'down': 0.0,
      'left': 0.0,
      'right': 0.0
    }
    self.moveDirection = {
      'up': False,
      'down': False,
      'left': False,
      'right': False
    }

  def move(self, type_str):
    self.moveDirection[type_str] = True
  
  def update(self):
    if (self.moveDirection['up']) {
      self.acceleration['up'] = 10
    }
    if (self.moveDirection['down']) {
      self.acceleration['down'] = 10
    }
    if (self.moveDirection['left']) {
      self.acceleration['left'] = 10
    }
    if (self.moveDirection['right']) {
      self.acceleration['right'] = 10
    }

    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction

    self.position['x'] = max(min(self.position['x'] + self.velocity['x'], 0), 8192)
    self.position['y'] = max(min(self.position['y'] + self.velocity['y'], 0), 8192)


class Deip(GameObject):
  def __init__(self, x, y):
    GameObject.__init__(self, x, y)
  
class Stuff(GameObject):
  def __init__(self, x, y, type):
    GameObject.__init__(self, x, y)

    reader = open('stuffType.json', 'r')
    stuff_type = json.loads(reader.read())
    type_number = random.randint(1, 5)
    self.attr = {
      'hp': stuff_type[str(type_number)]['HP'],
      'exp': stuff_type[str(type_number)]['EXP'],
      'body_damage': stuff_type[str(type_number)]['BodyDamage']
    }
  
  def update(self):
    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction

    self.position['x'] = max(min(self.position['x'] + self.velocity['x'], 0), 8192)
    self.position['y'] = max(min(self.position['y'] + self.velocity['y'], 0), 8192)


class Bullet(GameObject):
  def __init__(self, x, y, existence, damage):
    GameObject.__init__(self, x, y)

    self.attr = {
      'existence': existence,
      'damage': damage
    }
  
  def update():
    self.velocity['x'] = (self.acceleration['down'] - self.acceleration['up']) * friction
    self.velocity['y'] = (self.acceleration['right'] - self.acceleration['left']) * friction

    self.position['x'] = max(min(self.position['x'] + self.velocity['x'], 0), 8192)
    self.position['y'] = max(min(self.position['y'] + self.velocity['y'], 0), 8192)
