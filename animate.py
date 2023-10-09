################################################################################
# conway.py
#
# Author: electronut.in
#
# Description:
#
# A simple Python/matplotlib implementation of Conway's Game of Life.
################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use("agg")

with open("build_debug/log.txt", 'r') as fp:
    lines = len(fp.readlines())
    print('Total Number of lines:', lines)

f = open("assets/den312d.map", "r")
l = open("build_debug/log.txt", "r")
f.readline()
height = (int)(f.readline().split(" ")[1])
width = (int)(f.readline().split(" ")[1])
f.readline()

agents = (int)(l.readline().split("=")[1])


ON = 255
OFF = 0
vals = [ON, OFF]

# populate grid with random on/off - more off than on
grid = np.zeros((height, width))
print(grid.shape)

for i in range(2,16):
  if i==14:
    coords = l.readline().split("=")[1]
    coords = coords.split("),")
    for a in range(agents):
      splitVal = coords[a].split(",")
      r = (int)(splitVal[0][1:])
      c = (int)(splitVal[1])
      grid[c,r] = 200
  else:
    print(l.readline())

for r in range(height):
  row = f.readline()
  for c, ch in enumerate(row):
    if(c>=width):
      continue
    if ch=='T' or ch=='@':
      grid[r,c] = ON

original_grid = grid.copy()

count = 0

def update(data):
  global grid
  global count
  newGrid = original_grid.copy()
  coords = l.readline().split(":")[1]
  coords = coords.split("),")
  for a in range(agents):
    splitVal = coords[a].split(",")
    r = (int)(splitVal[0][1:])
    c = (int)(splitVal[1])
    newGrid[c,r] = 140

  # update data
  mat.set_data(newGrid)
  grid = newGrid
  return [mat]

# set up animation
fig, ax = plt.subplots()
mat = ax.matshow(grid)
ani = animation.FuncAnimation(fig, update, interval=lines-16,
                              save_count=lines-16)
# plt.savefig('foo.png')
ani.save('animation.mp4')

