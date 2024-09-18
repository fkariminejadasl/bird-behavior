import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""
Make a movie from a IMU accelaration.
"""

# fmt: off
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
# fmt: on
# Define the 3D rectangle vertices (a cuboid in this case)
rect_vertices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

# Read rotation angles from a CSV file
angles_df = pd.read_csv("/home/fatemeh/Downloads/bird/manouvre.csv", header=None)

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# # Set axis limits to [-1, 1]
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

# Set axis labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Manually set the color of axis labels
ax.xaxis.label.set_color("red")
ax.yaxis.label.set_color("green")
ax.zaxis.label.set_color("blue")


# Function to update the plot at each frame for the animation
def update(frame):
    # Get the rotation angles for the current frame from the CSV
    x_deg = math.degrees(angles_df[4].iloc[frame])
    y_deg = math.degrees(angles_df[5].iloc[frame])
    z_deg = math.degrees(angles_df[6].iloc[frame])

    ax.cla()  # Clear the axis to remove old cuboid

    # Rotate the X, Y, Z axes using the same rotation matrices
    rotated_axes = rotate_vertices(axes_vectors, x_deg, y_deg, z_deg)

    # fmt: off
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
    # fmt: on

    # Set the aspect ratio to auto
    ax.set_box_aspect([1, 1, 1])

    return ax


# Create the animation
ani = FuncAnimation(fig, update, frames=len(angles_df), interval=50)

# Save the animation to HTML
ani.save(filename="/home/fatemeh/Downloads/bird/html_example2.html", writer="html")
# ani.save("rotating_rectangle.mp4", writer='ffmpeg', fps=20)

# # Show the animation
plt.show()


# ax.view_init(elev=20, azim=30)
# import matplotlib
# matplotlib.use('TkAgg')
# from IPython.display import HTML
# # Convert the animation to HTML5 video
# html_video = ani.to_html5_video()
# # Display the video (useful in Jupyter Notebooks)
# HTML(html_video)
print("test")
