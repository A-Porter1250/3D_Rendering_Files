import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
# import Excel_Import as xlsx
import os

# function that creates the beam rendering
def model_beam(length, width, height, wall_thickness, data):
   
    # Define the origin of the outer face of the beam
    original_vertices_positions = np.array([
            [0, 0, 0],
            [0, width, 0],
            [0, width, height],
            [0, 0, height],
            [0, wall_thickness, wall_thickness],
            [0, width - wall_thickness, wall_thickness],
            [0, width - wall_thickness, height - wall_thickness],
            [0, wall_thickness, height - wall_thickness]
     ])
   
    # print("Original Vertices Array: ", original_vertices_positions)
    # Set an array that will be appended with the position of all the points (original with change by x value)

    num_rows = data.shape[0]
    vertices_array = np.zeros((num_rows,8,3))

    # Iterate the append commend for every x value in the data frame
    for i in range(num_rows):
        # Define a constant variable the same size as the original_veritces_position array
        points_add = original_vertices_positions.copy()
        # print(points_add)
        # Add to its length the x value of the ith point in the data frame
        points_add[:, 0] += data[i, 0]
        # Add to its height the height value at the ith row of the data frame
        points_add[:, 2] += data[i, 1]
        # Append the [8, 3] array to the defined
        vertices_array[i,:,:] = points_add
        # print(points_add)

    # Define the face array for the head of the beam
    head_face_vertices = np.array([
        [vertices_array[0, 0, :], vertices_array[0, 1, :], vertices_array[0, 2, :], vertices_array[0, 6, :], vertices_array[0, 5, :], vertices_array[0, 4, :]],
        [vertices_array[0, 0, :], vertices_array[0, 3, :], vertices_array[0, 2, :], vertices_array[0, 6, :], vertices_array[0, 7, :], vertices_array[0, 4, :]]
    ])

    outside_faces = np.zeros((4, 4, 3))
    inside_faces = np.zeros((4, 4, 3))
    
    for i in range(num_rows-1):
        # Define a place holder face variable
        for y in range(8):
            j=0
            Current_outside_face = np.array([
                [vertices_array[i,j,:], vertices_array[i+1,j,:], vertices_array[i+1,j+1,:], vertices_array[i,j+1,:]],
                [vertices_array[i,j+1,:], vertices_array[i+1,j+1,:], vertices_array[i+1,j+2,:], vertices_array[i,j+2,:]],
                [vertices_array[i,j+2,:], vertices_array[i+1,j+2,:], vertices_array[i+1,j+3,:], vertices_array[i,j+3,:]],
                [vertices_array[i,j+3,:], vertices_array[i+1,j+3,:], vertices_array[i+1,j+0,:], vertices_array[i,j+0,:]]
            ])
            outside_faces = np.vstack([outside_faces, Current_outside_face])  
            Current_inside_face = np.array([
                [vertices_array[i,j+4,:], vertices_array[i+1,j+4,:], vertices_array[i+1,j+5,:], vertices_array[i,j+5,:]],
                [vertices_array[i,j+5,:], vertices_array[i+1,j+5,:], vertices_array[i+1,j+6,:], vertices_array[i,j+6,:]],
                [vertices_array[i,j+6,:], vertices_array[i+1,j+6,:], vertices_array[i+1,j+7,:], vertices_array[i,j+7,:]],
                [vertices_array[i,j+7,:], vertices_array[i+1,j+7,:], vertices_array[i+1,j+4,:], vertices_array[i,j+4,:]]
            ])  
            inside_faces = np.vstack([inside_faces, Current_inside_face])

    print(length)
    back_face_vertices = np.array([
        [vertices_array[num_rows-1, 0, :], vertices_array[num_rows-1, 1, :], vertices_array[num_rows-1, 2, :], vertices_array[num_rows-1, 6, :], vertices_array[num_rows-1, 5, :], vertices_array[num_rows-1, 4, :]],
        [vertices_array[num_rows-1, 0, :], vertices_array[num_rows-1, 3, :], vertices_array[num_rows-1, 2, :], vertices_array[num_rows-1, 6, :], vertices_array[num_rows-1, 7, :], vertices_array[num_rows-1, 4, :]]
    ])
    print(back_face_vertices)


    # Plot the vertices_array
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax2.set_xlim(0, length)
    ax2.set_ylim(0, width)
    ax2.set_zlim(0, height)

    # Add vertices_array to the plot
    ax2.add_collection3d(Poly3DCollection(head_face_vertices, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.6))
    ax2.add_collection3d(Poly3DCollection(outside_faces[4:,:,:], facecolors='blue', linewidths=0.1, edgecolors='r', alpha=0.6))
    ax2.add_collection3d(Poly3DCollection(inside_faces[4:,:,:], facecolors='red', linewidths=0.1, edgecolors='k', alpha=0.1))
    ax2.add_collection3d(Poly3DCollection(back_face_vertices, facecolors='green', linewidths=1, edgecolors='r', alpha=0.6))
    

    # Show the plot
    plt.show()

'''
os.chdir(r"C:\\Users\\KIT08664\\Desktop")
data = pd.read_csv("Mechanics_Analysis_V2.csv")
array_data = data.iloc[:,[0,7]]
array_data = np.array(array_data)
print(array_data[0,0])
'''


data = np.column_stack((np.arange(0, 3, 0.05), np.zeros(60)))
# print(data)

# Call the model_beam function with example dimensions
model_beam(3, 4, 4, 0.5, data)