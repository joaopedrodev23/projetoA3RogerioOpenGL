import glfw
import glm

from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as gls

from PIL import Image

import tkinter as tk
from tkinter import filedialog

from geometry_utils import *
from FBX_utils import *


field_of_view = 60
window_size = [600, 600]

camera_pos = np.array([0, 0, 10], dtype=np.float32)
camera_aim = np.array([0, 0, 0], dtype=np.float32)
camera_up = np.array([0, 0, 1], dtype=np.float32)
far = 1000.0

shader_gouraud_shading_program = None
shader_phong_shading_program = None

fbx_model = None
current_shading_mode = 0

keys_used = {}


# ################################################################################################
# GLSL (Shaders) - Funções auxiliares
# ################################################################################################

def create_shader_program(vertex_shader_filepath, fragment_shader_filepath):
    # ler o código do vertex shader no caso de estar em um arquivo separado
    vertex_shader_source = ''
    with open(vertex_shader_filepath, 'r') as file:
        vertex_shader_source = file.read()

    # criar o objeto vertex shader
    vertex_shader = gls.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    
    # ler o código do fragment shader no caso de estar em um arquivo separado
    fragment_shader_source = ''
    with open(fragment_shader_filepath, 'r') as file:
        fragment_shader_source = file.read()

    # criar o objeto fragment shader
    fragment_shader = gls.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

    # criar o programa com os shaders
    shader_program = gls.compileProgram(vertex_shader, fragment_shader)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def create_vbo(buffer_type, buffer, index):
    vbo_id = glGenBuffers(1)
    glBindBuffer(buffer_type, vbo_id)
    glBufferData(buffer_type,     # tipo do buffer
                 buffer.nbytes,   # tamanho do buffer
                 buffer,          # dados
                 GL_STATIC_DRAW)  # forma do uso do buffer
    
    glVertexAttribPointer(index,              # código do atributo posição
                          buffer.shape[1],    # 2D ou 3D
                          GL_FLOAT,           # tipo dos valores do atributo
                          GL_FALSE,           # não desejo normalizar
                          0,                  # 2 floats de 4 bytes - quantidade de bytes entre um atributo e o próximo
                          ctypes.c_void_p(0)) # 0 pois começa no inicio do buffer (offset)
    glEnableVertexAttribArray(index)

    glBindBuffer(buffer_type, 0)


def create_vao(buffers):
    vao_id = glGenVertexArrays(1)
    glBindVertexArray(vao_id)

    for buffer in buffers:
        buffer_data = buffer[0]
        buffer_index = buffer[1]

        create_vbo(GL_ARRAY_BUFFER, buffer_data, buffer_index)

    glBindVertexArray(0)

    return vao_id


# ################################################################################################
# OpenGL - Funções auxiliares
# ################################################################################################

def get_vertices_lists_flat_shading(vertices, vertices_uvs, faces, faces_normals):
    vertex_pos_list = []
    vertex_normals_list = []
    vertex_uvs_list = []

    for i in range(len(faces)):
        face = faces[i]
        face_normal = faces_normals[i]

        for j in range(3):
            vertex_index = face[j][0]

            vertex = vertices[vertex_index]
            vertex_uv = vertices_uvs[vertex_index]

            vertex_pos_list.append(vertex)
            vertex_normals_list.append(face_normal)
            vertex_uvs_list.append(vertex_uv)

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32), np.array(vertex_uvs_list, dtype=np.float32)


def get_vertices_lists_vertex_shading(vertices, vertices_uvs, faces, vertices_normals):
    vertex_pos_list = []
    vertex_normals_list = []
    vertex_uvs_list = []

    for i in range(len(faces)):
        face = faces[i]

        for j in range(3):
            vertex_index = face[j][0]
            
            vertex = vertices[vertex_index]
            vertex_normal = vertices_normals[vertex_index]
            vertex_uv = vertices_uvs[vertex_index]

            vertex_pos_list.append(vertex)
            vertex_normals_list.append(vertex_normal)
            vertex_uvs_list.append(vertex_uv)   

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32), np.array(vertex_uvs_list, dtype=np.float32)


def compute_fbx_model_bounding_box(fbx_model):
    if len(fbx_model) == 0:
        return None

    fbx_model_bounding_box = compute_bounding_box(fbx_model[0][0], fbx_model[0][3])
   
    for i in range(1, len(fbx_model)):
        mesh = fbx_model[i]
        mesh_bounding_box = compute_bounding_box(mesh[0], mesh[3])

        fbx_model_bounding_box = union_bounding_boxes(fbx_model_bounding_box, mesh_bounding_box)

    return fbx_model_bounding_box


def load_texture(texture_paths):
    if len(texture_paths) == 0:
        return None
    
    final_image = Image.open(texture_paths[0])
    # for i in range(1, len(texture_paths)):
    #     next_image = Image.open(texture_paths[i])
    #     final_image = Image.blend(final_image.convert("RGBA"), next_image.convert("RGBA"), alpha=0.5)

    texture = final_image.transpose(Image.FLIP_TOP_BOTTOM)
    texture_data = np.array(texture.getdata(), np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.width, texture.height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

    # Empacotamento da textura (para formar ladrilhos por exemplo).
    # Só se vê o resultado se mapear os vértices para coordenadas s, t fora do intervalo [0, 1]
    # opções: GL_REPEAT (default), GL_MIRRORED_REPEAT e GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # Filtros de interpolação para o mapeamento (minificação - vários texels para um pixel / magnificação - um texel para vários pixels)
    # opções: GL_LINEAR e GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    
    # Aplicação da textura
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)

    return texture_id


def my_init(fbx_file_path):
    glClearColor(1, 1, 1, 1)
    glEnable(GL_DEPTH_TEST)
    
    # carrega o modelo
    global fbx_model
    fbx_model = load_fbx_model(fbx_file_path)
    bounding_box = compute_fbx_model_bounding_box(fbx_model)

    # calcula posição inicial da camera
    global camera_pos, camera_aim, camera_up, far
    camera_pos, camera_aim, camera_up, far = compute_camera_position(bounding_box, field_of_view)

    # compila os programas shaders:
    # 1 - um que implementa o shading de Gouraud, implementado pelo OpenGL Legado
    # 2 - outro que implementa o shading de Phong
    global shader_gouraud_shading_program
    shader_gouraud_shading_program = create_shader_program('04 - fbx_viewer_light_shader_gouraud_vs.glsl', '04 - fbx_viewer_light_shader_gouraud_fs.glsl')

    global shader_phong_shading_program
    shader_phong_shading_program = create_shader_program('04 - fbx_viewer_light_shader_phong_vs.glsl', '04 - fbx_viewer_light_shader_phong_fs.glsl')

    # cria dois VAOs: 
    # 1 - um para flat shading com a mesma normal para os 3 vértices de uma mesma face e
    # 2 - outro para gouraud e phong shading com a normal calculada por vértice como sendo a média das normais das faces em torno do vértice
    for mesh in fbx_model:
        vertices_pos, vertices_normals, vertices_uvs, faces, faces_normals, texture_paths = mesh

        mesh_vertices_pos, mesh_vertices_normals, mesh_vertices_uvs = get_vertices_lists_flat_shading(vertices_pos, vertices_uvs, faces, faces_normals)
        vao_flat_shading_id = create_vao([[mesh_vertices_pos, 0], [mesh_vertices_normals, 1], [mesh_vertices_uvs, 2]])

        mesh_vertices_pos, mesh_vertices_normals, mesh_vertices_uvs = get_vertices_lists_vertex_shading(vertices_pos, vertices_uvs, faces, vertices_normals)
        vao_vertex_shading_id = create_vao([[mesh_vertices_pos, 0], [mesh_vertices_normals, 1], [mesh_vertices_uvs, 2]])

        texture_id = load_texture(texture_paths)

        mesh.append(vao_flat_shading_id)
        mesh.append(vao_vertex_shading_id) 
        mesh.append(texture_id)
        mesh.append(len(mesh_vertices_pos))


def get_current_vao_index():
    if current_shading_mode % 3 == 0:
        return -4
    
    if current_shading_mode % 3 == 1:
       return -3

    return -3


def get_current_shader_program():
    if current_shading_mode % 3 == 0:
        return shader_gouraud_shading_program
    
    if current_shading_mode % 3 == 1:
       return shader_gouraud_shading_program

    return shader_phong_shading_program


def my_render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_size[0], window_size[1])

    current_shader_program = get_current_shader_program()
    glUseProgram(current_shader_program)

    # Matrizes
    model = glm.mat4(1.0)
    view = glm.lookAt(glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2]), 
                      glm.vec3(camera_aim[0], camera_aim[1], camera_aim[2]), 
                      glm.vec3(camera_up[0], camera_up[1], camera_up[2]))    
    projection = glm.perspective(glm.radians(field_of_view), float(window_size[0])/float(window_size[1]), 0.01, far)
    light_pos = camera_pos + camera_up * 10

    # Uniforms de transformação para o vertex shader
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    # Uniforms de iluminação para o fragment shader
    glUniform3f(glGetUniformLocation(current_shader_program, "objectColor"), 1.0, 1.0, 1.0)
    glUniform3fv(glGetUniformLocation(current_shader_program, "viewPos"), 1, camera_pos)

    glUniform3f(glGetUniformLocation(current_shader_program, "ambientColor"), 0.3, 0.3, 0.3)
    glUniform3f(glGetUniformLocation(current_shader_program, "lightColor"), 1.0, 1.0, 1.0)
    glUniform3fv(glGetUniformLocation(current_shader_program, "lightPos"), 1, light_pos)
    glUniform1f(glGetUniformLocation(current_shader_program, "shininess"), 64.0)

    current_vao_index = get_current_vao_index()

    for mesh in fbx_model:
        glBindVertexArray(mesh[current_vao_index])

        texture_id = mesh[-2]
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(current_shader_program, "bindTexture"), 0)

        glDrawArrays(GL_TRIANGLES, 0, mesh[-1])

        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)

    glUseProgram(0)


# ################################################################################################
# GLFW - Funções auxiliares
# ################################################################################################

def my_update_window_size(window, width, height):
    global window_size
    window_size = [width, height]


def rotate_view_pos_horizontal(angle_rotate_rads):
    global camera_pos

    view_direction = np.array(camera_pos) - np.array(camera_aim)
    side_direction = np.cross(view_direction, np.array([0.0, 1.0, 0.0]))
    up = np.cross(side_direction, view_direction)
    up = up / np.linalg.norm(up)

    rotate_matrix = get_rotate_matrix(angle_rotate_rads, up)

    camera_pos = np.dot(view_direction, rotate_matrix.T) + np.array(camera_aim)
    camera_pos = camera_pos.tolist()


def rotate_view_pos_vertical(angle_rotate_rads):
    global camera_pos

    view_direction = np.array(camera_pos) - np.array(camera_aim)
    side_direction = np.cross(view_direction, np.array([0.0, 1.0, 0.0]))
    side_direction = side_direction / np.linalg.norm(side_direction)

    rotate_matrix = get_rotate_matrix(angle_rotate_rads, side_direction)

    camera_pos = np.dot(view_direction, rotate_matrix.T) + np.array(camera_aim)
    camera_pos = camera_pos.tolist()


def my_keyboard(window, key, scancode, action, mods):
    global keys_used, current_shading_mode

    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    if action == glfw.PRESS and key == glfw.KEY_F12:
        current_shading_mode = current_shading_mode + 1

    if action == glfw.REPEAT or action == glfw.PRESS:
        keys_used[key] = True
    elif action == glfw.RELEASE:
        keys_used[key] = False


def process_user_interaction(window):
    angle_rotate_rads = 0.08

    if keys_used.get(glfw.KEY_LEFT, False) or keys_used.get(glfw.KEY_A, False):
        rotate_view_pos_horizontal(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_RIGHT, False) or keys_used.get(glfw.KEY_D, False):
        rotate_view_pos_horizontal(angle_rotate_rads)

    if keys_used.get(glfw.KEY_UP, False) or keys_used.get(glfw.KEY_W, False):
        rotate_view_pos_vertical(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_DOWN, False) or keys_used.get(glfw.KEY_S, False):
        rotate_view_pos_vertical(angle_rotate_rads)


# ################################################################################################
# Main - Programa Principal
# ################################################################################################

def open_fbx_file():
    # Cria janela "oculta"
    root = tk.Tk()
    root.withdraw()  

    # Abre diálogo de seleção
    arquivo = filedialog.askopenfilename(
    title="Selecione um arquivo",
        filetypes=(("Arquivos modelos fbx", "*.fbx"), 
                   ("Todos os arquivos", "*.*"))
    )

    return arquivo

def main():
    fbx_file_path = open_fbx_file()
    if not open_fbx_file:
        return

    glfw.init()

    window = glfw.create_window(window_size[0], window_size[1], "FBX Viewer", None, None)
    glfw.make_context_current(window) 
    glfw.swap_interval(1)

    glfw.set_framebuffer_size_callback(window, my_update_window_size)
    glfw.set_key_callback(window, my_keyboard)

    glfw.maximize_window(window)

    my_init(fbx_file_path)

    while not glfw.window_should_close(window):
        glfw.poll_events() # escutando eventos
        process_user_interaction(window)

        my_render()

        glfw.swap_buffers(window) # alternando os buffers

    glfw.terminate()


if __name__ == "__main__":
    main()