import numpy as np


def load_obj_file(obj_file_path):
    vertices_pos = []
    vertices_normals = []
    vertices_texcoords = []
    faces = []

    lines = []
    with open(obj_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            vertices_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('vn '):
            parts = line.split()
            vertices_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('vt '):
            parts = line.split()
            vertices_texcoords.append([float(parts[1]), float(parts[2])])
        elif line.startswith('f '):
            parts = line.split()
            face_indices = []
            for p in parts[1:]:
                # Handle different face formats (e.g., v, v/vt, v/vt/vn, v//vn)
                indices = [int(i) - 1 for i in p.split('/') if i] # -1 for 0-based indexing
                face_indices.append(indices)
            faces.append(face_indices)

    return np.array(vertices_pos, dtype=np.float32), \
        np.array(vertices_normals, dtype=np.float32), \
        np.array(vertices_texcoords, dtype=np.float32), \
        np.array(faces, dtype=np.uint32)


def compute_bounding_box(vertices_pos, faces):
    """
    Alguns arquivos OBJs contem vertices não utilizados pelas faces.
    Então, primeiro gerei uma coleção com os vértices utilizados.
    
    Cada bbox é uma lista com dois pontos:
        [ [xmin, ymin, zmin], [xmax, ymax, zmax] ]
    """

    # Converter para arrays NumPy
    faces = np.asarray(faces, dtype=np.int32)
    vertices_pos = np.asarray(vertices_pos, dtype=np.float32)

    # Caso as faces tenham estrutura [[[a], [b], [c]]], achatamos o eixo extra
    if faces.ndim == 3:
        faces = faces[..., 0]

    # Extrai os vértices usados diretamente
    used_vertices = vertices_pos[faces.ravel()]

    xv, yv, zv = zip(*used_vertices)

    min_x, max_x = min(xv), max(xv)
    min_y, max_y = min(yv), max(yv)
    min_z, max_z = min(zv), max(zv)

    return np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]], dtype=np.float32)


def union_bounding_boxes(bbox1, bbox2):
    """
    Calcula a união de dois bounding boxes 3D.
    
    Cada bbox é uma lista com dois pontos:
        [ [xmin, ymin, zmin], [xmax, ymax, zmax] ]
    """
    bbox1_min, bbox1_max = bbox1
    bbox2_min, bbox2_max = bbox2

    bbox_min = np.minimum(bbox1_min, bbox2_min) 
    bbox_max = np.maximum(bbox1_max, bbox2_max)

    return [bbox_min, bbox_max]


def translate_bounding_box(bbox, displacement):
    """
    Translada um bounding box 3D pelo vetor displacement: [dx, dy, dz].
    
    bbox: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    Retorna um novo bbox transladado.
    """
    bbox_min, bbox_max = bbox

    return [bbox_min + displacement, bbox_max + displacement]


def get_bounding_box_center(bbox):
    """
    Calcula o centro de um bounding box 3D.
    
    bbox: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    Retorna uma lista [cx, cy, cz] com o centro.
    """
    bbox_min, bbox_max = bbox
    center = (bbox_min + bbox_max) / 2.0

    return center


def compute_faces_normals(vertices_pos, faces):
    # Converte faces para array (N, 3, 1)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    
    # Extrai as posições dos 3 vértices de cada face
    p0 = vertices_pos[faces[:, 0, ...] if faces.ndim == 3 else faces[:, 0]]
    p1 = vertices_pos[faces[:, 1, ...] if faces.ndim == 3 else faces[:, 1]]
    p2 = vertices_pos[faces[:, 2, ...] if faces.ndim == 3 else faces[:, 2]]

    # Calcula vetores das arestas
    u = p1 - p0
    w = p2 - p0

    # Produto vetorial (normal bruta de cada face)
    face_normals = np.cross(u, w)

    # Normaliza todas de uma vez
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / (norms + 1e-9)

    return face_normals.astype(np.float32)


def compute_vertices_normals(vertices_pos, faces):
    # Garante que faces seja um array Nx3 de índices inteiros
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    # Calcula normais das faces (vetorizado)
    p0 = vertices_pos[faces[:, 0]]
    p1 = vertices_pos[faces[:, 1]]
    p2 = vertices_pos[faces[:, 2]]

    face_normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / (norms + 1e-9)

    # Inicializa acumulador
    vertices_normals = np.zeros_like(vertices_pos)

    # Soma as normais de cada face nos vértices correspondentes
    np.add.at(vertices_normals, faces[:, 0], face_normals)
    np.add.at(vertices_normals, faces[:, 1], face_normals)
    np.add.at(vertices_normals, faces[:, 2], face_normals)

    # Normaliza os vetores finais
    norms = np.linalg.norm(vertices_normals, axis=1, keepdims=True)
    vertices_normals = vertices_normals / (norms + 1e-9)

    return vertices_normals.astype(np.float32)


def compute_camera_position(bouding_box, fov_y_deg=45.0, aspect_ratio=1.0, up=[0,1,0]):
    """
    Calcula a posição da câmera para visualizar o modelo dentro do bounding box.

    bbox_min : (x_min, y_min, z_min)
    bbox_max : (x_max, y_max, z_max)
    fov_y_deg : campo de visão vertical em graus
    aspect_ratio : proporção largura/altura da tela
    up : vetor 'up' da câmera
    """

    bbox_min, bbox_max = bouding_box

    # Centro do bounding box
    bbox_center = get_bounding_box_center(bouding_box)
    
    # Raio da esfera que engloba o bbox
    radius = np.linalg.norm(bbox_max - bbox_center)

    # Converte FOV para radianos
    fov_y = np.radians(fov_y_deg)

    # Distância necessária da câmera para caber o objeto na vertical
    dist_y = radius / np.sin(fov_y / 2.0)

    # Ajuste também considerando o aspecto (horizontal)
    fov_x = 2.0 * np.arctan(np.tan(fov_y/2.0) * aspect_ratio)
    dist_x = radius / np.sin(fov_x / 2.0)

    # Pega a maior distância necessária
    distance = max(dist_x, dist_y)

    # Define posição da câmera olhando no -Z (padrão OpenGL clássico)
    camera_pos = bbox_center + np.array([0, 0, distance], dtype=np.float32)

    far = 3 * distance

    return camera_pos, bbox_center, np.array(up), far


def get_rotate_matrix(angle_rotate_rads, rotate_axis):
    cos_angle = np.cos(angle_rotate_rads)
    sin_angle = np.sin(angle_rotate_rads)

    # Matriz de rotação usando o Teorema de Rodrigues
    rotate_matrix = np.array([
        [cos_angle + rotate_axis[0]**2 * (1 - cos_angle), rotate_axis[0] * rotate_axis[1] * (1 - cos_angle) - rotate_axis[2] * sin_angle, rotate_axis[0] * rotate_axis[2] * (1 - cos_angle) + rotate_axis[1] * sin_angle],
        [rotate_axis[1] * rotate_axis[0] * (1 - cos_angle) + rotate_axis[2] * sin_angle, cos_angle + rotate_axis[1]**2 * (1 - cos_angle), rotate_axis[1] * rotate_axis[2] * (1 - cos_angle) - rotate_axis[0] * sin_angle],
        [rotate_axis[2] * rotate_axis[0] * (1 - cos_angle) - rotate_axis[1] * sin_angle, rotate_axis[2] * rotate_axis[1] * (1 - cos_angle) + rotate_axis[0] * sin_angle, cos_angle + rotate_axis[2]**2 * (1 - cos_angle)]
    ], dtype=np.float32)

    return rotate_matrix
