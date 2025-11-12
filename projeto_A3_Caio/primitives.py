import math
import random

from geometry_utils import *

def get_cube_model(color=[1, 0, 0], color_random=False):
    cube_vertices = [
        [-0.5,  0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5]
    ]

    cube_faces = [
        [[0], [1], [2], [3]],
        [[3], [2], [6], [7]],
        [[7], [6], [5], [4]],
        [[4], [5], [1], [0]],
        [[0], [3], [7], [4]],
        [[5], [6], [2], [1]]
    ]

    cube_faces_uvs = [
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1]],
    ]

    cube_faces_color = []
    for i in range(len(cube_faces)):
        if color_random:
            cube_faces_color.append([random.random(), random.random(),  random.random()])
        else:
            cube_faces_color.append(color)

    cube_vertices = np.array(cube_vertices, dtype=np.float32)
    cube_faces = np.array(cube_faces, dtype=np.int32)
    cube_faces_color = np.array(cube_faces_color, dtype=np.float32)
    cube_faces_uvs = np.array(cube_faces_uvs, dtype=np.float32) 

    cube_faces_normals = compute_faces_normals(cube_vertices, cube_faces)

    return cube_vertices, cube_faces, cube_faces_normals, cube_faces_color, cube_faces_uvs


def get_cube_model_triangles(color=[1, 0, 0], color_random=False):
    cube_vertices = [
        [-0.5,  0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5]
    ]

    cube_faces = [
        [[0], [1], [2]],
        [[0], [2], [3]],
        [[3], [2], [6]],
        [[3], [6], [7]],
        [[7], [6], [5]],
        [[7], [5], [4]],
        [[4], [5], [1]],
        [[4], [1], [0]],
        [[0], [3], [7]],
        [[0], [7], [4]],
        [[5], [6], [2]],
        [[5], [2], [1]]
    ]

    cube_faces_uvs = [
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 1], [1, 0], [1, 1]]
    ]

    cube_faces_color = []
    for _ in range(len(cube_faces) // 2):
        if color_random:
            cube_faces_color.append([random.random(), random.random(),  random.random()])
        else:
            cube_faces_color.append(color)

        cube_faces_color.append(cube_faces_color[-1])

    cube_vertices = np.array(cube_vertices, dtype=np.float32)
    cube_faces = np.array(cube_faces, dtype=np.int32)
    cube_faces_color = np.array(cube_faces_color, dtype=np.float32)
    cube_faces_uvs = np.array(cube_faces_uvs, dtype=np.float32) 

    cube_faces_normals = compute_faces_normals(cube_vertices, cube_faces)

    return cube_vertices, cube_faces, cube_faces_normals, cube_faces_color, cube_faces_uvs


def get_sphere_model(sectors=60, stacks=20, color=[1, 0, 0], color_random=False):
    sector_step = 2 * math.pi / sectors 
    stack_step = math.pi / stacks

    sphere_vertices = []

    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * stack_step  # de pi/2 a -pi/2
        xz = math.cos(stack_angle)  # r * cos(u)
        y = math.sin(stack_angle)   # r * sin(u)

        for j in range(sectors + 1):
            sector_angle = j * sector_step  # de 0 a 2pi

            # posição do vértice (x, y, z)
            x = xz * math.sin(sector_angle)  # r * cos(u) * sin(v)
            z = xz * math.cos(sector_angle)  # r * cos(u) * cos(v)
            sphere_vertices.append([x, y, z])

    sphere_faces = []
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1

        for j in range(sectors):

            if i != 0:
                sphere_faces.append([k1, k2, k1 + 1])

            if i != (stacks -1):
                sphere_faces.append([k1 + 1, k2, k2 + 1])

            k1 = k1 + 1
            k2 = k2 + 1

    sphere_faces_color = []
    for i in range(len(sphere_faces)):
        if color_random:
            sphere_faces_color.append([random.random(), random.random(),  random.random()])
        else:
            sphere_faces_color.append(color)

    sphere_vertices = np.array(sphere_vertices, dtype=np.float32)
    sphere_faces = np.array(sphere_faces, dtype=np.int32)   
    sphere_faces_color = np.array(sphere_faces_color, dtype=np.float32)

    sphere_faces_normals = compute_faces_normals(sphere_vertices, sphere_faces)

    return sphere_vertices, sphere_faces, sphere_faces_color, sphere_faces_normals
    