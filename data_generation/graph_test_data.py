import os
import json
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = 'database'

files = os.listdir(path)

x_points = []
y_points = []
z_points = []

x_delta = []
y_delta = []
z_delta = []

flags = []

for file_name in files[:1000]:

    total_path = os.path.join(path, file_name)

    with open(total_path, 'r') as file:
        data = json.load(file)

        x = data['t_x']
        y = data['t_y']
        z = data['t_z']

        x_points.append(x)
        y_points.append(y)
        z_points.append(z)

        vel = data['target_velocity']
        vel_head = data['target_velocity_heading']

        dx = vel * math.cos(math.radians(vel_head))
        dy = 0
        dz = vel * math.sin(math.radians(vel_head))

        x_delta.append(dx)
        y_delta.append(dy)
        z_delta.append(dz)

        d_before = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))
        d_after = math.sqrt(math.pow(x + dx, 2) + math.pow(y + dy, 2) + math.pow(z + dz, 2))

        flags.append(d_after < d_before)
    

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x, y, z, u, v, w, flag in zip(x_points, y_points, z_points, x_delta, y_delta, z_delta, flags):
    color = 'red' if flag else 'blue'
    ax.quiver(x, z, y, u, w, v, color=color, length=100, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.show()