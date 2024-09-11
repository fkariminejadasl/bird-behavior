import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
from matplotlib.animation import FuncAnimation

# Function to create a rotation matrix around X-axis
def rotation_matrix_x(degrees):
    radians = math.radians(degrees)
    return np.array([[1, 0, 0],
                     [0, math.cos(radians), -math.sin(radians)],
                     [0, math.sin(radians), math.cos(radians)]])

# Function to create a rotation matrix around Y-axis
def rotation_matrix_y(degrees):
    radians = math.radians(degrees)
    return np.array([[math.cos(radians), 0, math.sin(radians)],
                     [0, 1, 0],
                     [-math.sin(radians), 0, math.cos(radians)]])

# Function to create a rotation matrix around Z-axis
def rotation_matrix_z(degrees):
    radians = math.radians(degrees)
    return np.array([[math.cos(radians), -math.sin(radians), 0],
                     [math.sin(radians), math.cos(radians), 0],
                     [0, 0, 1]])

# Function to rotate vertices by given degrees on X, Y, and Z axes
def rotate_vertices(vertices, x_deg, y_deg, z_deg):
    # Apply the rotation matrices sequentially
    rotation_matrix = rotation_matrix_x(x_deg) @ rotation_matrix_y(y_deg) @ rotation_matrix_z(z_deg)
    return np.dot(vertices, rotation_matrix.T)

# Define the initial direction vectors of the X, Y, Z axes
axes_vectors = np.array([[1, 0, 0],  # X-axis direction
                         [0, 1, 0],  # Y-axis direction
                         [0, 0, 1]]) # Z-axis direction

# Define the 3D rectangle vertices (a cuboid in this case)
rect_vertices = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 1],
                          [1, 1, 1],
                          [0, 1, 1]])

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Manually set the color of axis labels
ax.xaxis.label.set_color('red')
ax.yaxis.label.set_color('green')
ax.zaxis.label.set_color('blue')


# Initial rotation angles
x_deg, y_deg, z_deg = 0, 0, 0

# Function to update the plot at each frame for the animation
def update(frame):
    global x_deg, y_deg, z_deg
    ax.cla()  # Clear the axis to remove old cuboid

    # Increment rotation angles for smooth animation
    x_deg += 2  # Rotate by 2 degrees in X
    y_deg += 2  # Rotate by 2 degrees in Y
    z_deg += 2  # Rotate by 2 degrees in Z

    # Rotate the X, Y, Z axes using the same rotation matrices
    rotated_axes = rotate_vertices(axes_vectors, x_deg, y_deg, z_deg)
    
    # Draw the rotated X, Y, Z axes
    ax.quiver(0, 0, 0, rotated_axes[0, 0], rotated_axes[0, 1], rotated_axes[0, 2], color='red', arrow_length_ratio=0.1)  # Rotated X-axis
    ax.quiver(0, 0, 0, rotated_axes[1, 0], rotated_axes[1, 1], rotated_axes[1, 2], color='green', arrow_length_ratio=0.1)  # Rotated Y-axis
    ax.quiver(0, 0, 0, rotated_axes[2, 0], rotated_axes[2, 1], rotated_axes[2, 2], color='blue', arrow_length_ratio=0.1)  # Rotated Z-axis

    # Rotate the rectangle
    rotated_vertices = rotate_vertices(rect_vertices, x_deg, y_deg, z_deg)

    # Define the cuboid faces using the rotated vertices
    faces = [[rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],  # bottom face
             [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],  # top face
             [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],  # front face
             [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],  # back face
             [rotated_vertices[0], rotated_vertices[3], rotated_vertices[7], rotated_vertices[4]],  # left face
             [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]]]  # right face

    # Draw the 3D rectangle (cuboid) using Poly3DCollection
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=0, alpha=.25))
    
    # Set the aspect ratio to auto
    ax.set_box_aspect([1,1,1])
    
    return ax

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
ani.save(filename="/home/fatemeh/Downloads/bird/html_example.html", writer="html")
# ani.save('/home/fatemeh/Downloads/bird/rotating_rectangle.mp4', writer='ffmpeg', fps=20)
# # Show the animation
plt.show()






# # Set axis labels without color
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# # Manually set the color of axis labels
# ax.xaxis.label.set_color('red')
# ax.yaxis.label.set_color('green')
# ax.zaxis.label.set_color('blue')

# ax.view_init(elev=20, azim=30)
# import matplotlib
# matplotlib.use('TkAgg')
print("test")
