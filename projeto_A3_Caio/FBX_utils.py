import numpy as np

import fbx
from FbxCommon import *

from geometry_utils import *

debug_fbx = False


def get_fbx_vertex_element_value(element, point_index, vertex_index):
    element_value = None

    mapping = element.GetMappingMode()
    reference = element.GetReferenceMode()

    if mapping == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
        if reference == fbx.FbxLayerElement.EReferenceMode.eDirect:
            element_value = element.GetDirectArray().GetAt(point_index)

        else:
            normal_index = element.GetIndexArray().GetAt(point_index)
            element_value = element.GetDirectArray().GetAt(normal_index)
    
    elif mapping == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
        if reference == fbx.FbxLayerElement.EReferenceMode.eDirect:
            element_value = element.GetDirectArray().GetAt(vertex_index)

        else:
            normal_index = element.GetIndexArray().GetAt(vertex_index)
            element_value = element.GetDirectArray().GetAt(normal_index)

    return element_value


def get_texture_filenames_from_material(material):
    texture_filenames = []

    material_properties = [
        fbx.FbxSurfaceMaterial.sDiffuse,
        # fbx.FbxSurfaceMaterial.sSpecular
    ]

    for prop_name in material_properties:
        material_property = material.FindProperty(prop_name)
        if material_property.IsValid():
            num_textures = material_property.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId))
            for i in range(num_textures):
                texture = material_property.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId), i)
                if texture:
                    if isinstance(texture, fbx.FbxFileTexture):
                        texture_filename = texture.GetFileName()
                        if texture_filename:
                            texture_filenames.append(texture_filename)

    return texture_filenames


def load_fbx_node_geometry(node):
    vertices_pos = []
    vertices_normals = []
    vertices_uvs = []
    faces = []

    mesh = node.GetMesh()
    if debug_fbx:
        print('\tMesh Name: ', mesh.GetName())
 
    normal_element = mesh.GetElementNormal()
    has_normals = normal_element is not None

    uv_element = mesh.GetElementUV()
    has_uvs = uv_element is not None

    control_points = mesh.GetControlPoints()

    num_polygons = mesh.GetPolygonCount()
    if debug_fbx:
        print('\tMesh num polygons: {0}'.format(num_polygons))

    vertex_index = 0
    for i in range(num_polygons):
        num_polygon_points = mesh.GetPolygonSize(i)

        if debug_fbx:
            print('\t\tPolygon {0}: {1} points'.format(i, num_polygon_points))

        face_vertices_pos = []
        face_vertices_normals = []
        face_vertices_uvs = []
        face = []

        polygon_indices = [mesh.GetPolygonVertex(i, j) for j in range(num_polygon_points)]
        for j in range(1, num_polygon_points - 1):
            triangle = [polygon_indices[0], polygon_indices[j], polygon_indices[j + 1]]
            
            for point_index in triangle:
                point = control_points[point_index]
                face_vertices_pos.append([point[0], point[1], point[2]])

                if has_normals:
                    normal = get_fbx_vertex_element_value(normal_element, point_index, vertex_index)
                    if normal is not None:
                        face_vertices_normals.append([normal[0], normal[1], normal[2]])
                    else:
                        print('Warning: normal is None for point_index {0}, vertex_index {1}'.format(point_index, vertex_index))
                        face_vertices_normals.append([0.0, 0.0, 0.0])

                if has_uvs:
                    uv = get_fbx_vertex_element_value(uv_element, point_index, vertex_index)
                    if uv is not None:
                        face_vertices_uvs.append([uv[0], uv[1]])
                    else:
                        print('Warning: uv is None for point_index {0}, vertex_index {1}'.format(point_index, vertex_index))
                        face_vertices_uvs.append([0.0, 0.0])

                vertex_index += 1

            face.append([[vertex_index - 3], [vertex_index - 2], [vertex_index - 1]])

        vertices_pos.extend(face_vertices_pos)
        vertices_normals.extend(face_vertices_normals)
        vertices_uvs.extend(face_vertices_uvs)
        faces.extend(face)

    texture_paths = []
    for i in range(node.GetMaterialCount()):
        material = node.GetMaterial(i)
        texture_paths.extend(get_texture_filenames_from_material(material))
    texture_paths = list(dict.fromkeys(texture_paths))

    vertices_pos = np.array(vertices_pos, dtype=np.float32)
    vertices_normals = np.array(vertices_normals, dtype=np.float32)
    vertices_uvs = np.array(vertices_uvs, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return vertices_pos, vertices_normals, vertices_uvs, faces, texture_paths


def load_fbx_geometry(scene):
    model = []

    root_node = scene.GetRootNode()

    if root_node:
        if debug_fbx:
            print('Root node num child: {0}'.format(root_node.GetChildCount()))

        for i in range(root_node.GetChildCount()):
            node = root_node.GetChild(i)
            attrib_type = node.GetNodeAttribute().GetAttributeType()
            if attrib_type != fbx.FbxNodeAttribute.EType.eMesh:
                continue

            node_vertices_pos, node_vertices_normals, node_vertices_uvs, node_faces, node_texture_paths = load_fbx_node_geometry(node)

            if len(node_vertices_pos) > 0:
                if (len(node_vertices_normals) == 0):
                    node_vertices_normals = compute_vertices_normals(node_vertices_pos, node_faces)

                node_faces_normals = compute_faces_normals(node_vertices_pos, node_faces)
                model.append([node_vertices_pos, node_vertices_normals, node_vertices_uvs, node_faces, node_faces_normals, node_texture_paths])

    return model


def load_fbx_model(filepath):
    # Prepare the FBX SDK & load the scene
    sdk_manager, scene = InitializeSdkObjects()
    result = LoadScene(sdk_manager, scene, filepath)
    if not result:
        print('An error occurred while loading the scene...')
        return None, None, None, None

    print('Load scene...')
    return load_fbx_geometry(scene)
