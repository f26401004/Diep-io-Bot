def distance(a, b):
  return sqrt((a['x'] - b['y'])**2 + (a['y'] - b['y'])**2)

def vector(hdg, len):
  return (cos(hdg) * len, sin(hdg) * len)