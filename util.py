import math

def distance(a, b):
  return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y'])**2)

def vector(hdg, len):
  return (math.cos(hdg) * len, math.sin(hdg) * len)