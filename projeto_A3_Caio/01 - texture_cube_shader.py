import glfw
import glm

from enum import Enum
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as gls

from PIL import Image

from geometry_utils import *
from primitives import *

window_size = [600, 600]

camera_pos = [0, 0, 10]
camera_aim = [0, 0, 0]
camera_up = [0, 1, 0]
field_of_view = 60

shader_program = None
scene = None

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

def create_scene():
    cube_vertices, cube_faces, cube_faces_normals, _, cube_faces_uvs = get_cube_model_triangles(color_random=True)
    
    minecraft_texture_id = load_texture('Textures/minecraft.jpg', 180)
    
    color_dice_texture_id = load_texture('Textures/dado_colorido.jpg', 0)
    # lado 1 - (0, 800), (224, 800), (224, 576), (0, 576) => [0, 0.78], [0.22, 0.78], [0.22, 0.56], [0, 0.56]
    # lado 2 - (224, 800), (448, 800), (448, 576), (224, 576) => [0.22, 0.78], [0.44, 0.78], [0.44, 0.56], [0.22, 0.56]
    # ...
    color_dice_faces_uvs = [
        [[0, 0.78],    [0.22, 0.78], [0.22, 0.56]],
        [[0, 0.78],    [0.22, 0.56], [0, 0.56]],
        [[0.22, 0.78], [0.44, 0.78], [0.44, 0.56]],
        [[0.22, 0.78], [0.44, 0.56], [0.22, 0.56]],
        [[0.44, 0.78], [0.66, 0.78], [0.66, 0.56]],
        [[0.44, 0.78], [0.66, 0.56], [0.44, 0.56]],
        [[0.66, 0.78], [0.88, 0.78], [0.88, 0.56]],
        [[0.66, 0.78], [0.88, 0.56], [0.66, 0.56]],
        [[0.44, 0.56], [0.66, 0.56], [0.66, 0.34]],
        [[0.44, 0.56], [0.66, 0.34], [0.44, 0.34]],
        [[0.44, 1],    [0.66, 1],    [0.66, 0.78]],
        [[0.44, 1],    [0.66, 0.78], [0.44, 0.78]],
    ]

    scene = [
        [cube_vertices, cube_faces, cube_faces_normals, None, None, [0, 0, 0], [0, 0, 1, 0], [1, 1, 1]],
        [cube_vertices, cube_faces, cube_faces_normals, minecraft_texture_id, cube_faces_uvs, [2, 0, 0], [0, 0, 1, 0], [1, 1, 1]],
        [cube_vertices, cube_faces, cube_faces_normals, color_dice_texture_id, color_dice_faces_uvs, [-2, 0, 0], [0, 0, 1, 0], [1, 1, 1]]
    ]

    return scene


def compute_scene_bounding_box(scene):
    if len(scene) == 0:
        return [[0, 0, 0], [0, 0, 0]]
    
    first_model = scene[0]
    scene_bounding_box = translate_bounding_box(compute_bounding_box(first_model[0], first_model[1]), first_model[5])

    for i in range(1, len(scene)):
        model = scene[i]
        model_bounding_box = translate_bounding_box(compute_bounding_box(model[0], model[1]), model[5])
        scene_bounding_box = union_bounding_boxes(scene_bounding_box, model_bounding_box)

    return scene_bounding_box


def load_texture(texture_filename, rotate_angle):
    texture = Image.open(texture_filename)
    texture = texture.rotate(rotate_angle)
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


def get_vertices_lists_flat_shading(vertices, faces, faces_normals, faces_uvs):
    vertex_pos_list = []
    vertex_normals_list = []
    vertex_uvs_list = []

    for i in range(len(faces)):
        face = faces[i]
        face_normal = faces_normals[i]
        face_uvs = faces_uvs[i] if faces_uvs is not None else None

        for j in range(3):
            vertex_index = face[j][0]
            
            vertex = vertices[vertex_index]
            vertex_pos_list.append(vertex)

            vertex_normals_list.append(face_normal)
            
            if faces_uvs is not None:
                vertex_uvs_list.append(face_uvs[j])
            else:
                vertex_uvs_list.append([0.0, 0.0])

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32), np.array(vertex_uvs_list, dtype=np.float32)


def my_init():
    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)

    # carrega a cena
    global scene
    scene = create_scene()
    bounding_box = compute_scene_bounding_box(scene)

    # calcula posição inicial da camera
    global camera_pos, camera_aim, camera_up, far
    camera_pos, camera_aim, camera_up, far = compute_camera_position(bounding_box, field_of_view)

    # compila o programa shader:
    global shader_program
    shader_program = create_shader_program('01 - texture_cube_shader_vs.glsl', '01 - texture_cube_shader_fs.glsl')

    # cria de cada modelo da cena
    for model in scene:
        vertices_pos = model[0]
        faces = model[1]
        faces_normals = model[2]
        faces_uvs = model[4]

        model_vertices_pos, model_vertices_normals, model_vertices_uvs = get_vertices_lists_flat_shading(vertices_pos, faces, faces_normals, faces_uvs)
        vao_flat_shading_id = create_vao([[model_vertices_pos, 0], [model_vertices_normals, 1], [model_vertices_uvs, 2]])

        model.append(vao_flat_shading_id)
        model.append(len(model_vertices_pos))


def my_render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_size[0], window_size[1])

    glUseProgram(shader_program)

    # Matrizes
    view = glm.lookAt(glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2]), 
                      glm.vec3(camera_aim[0], camera_aim[1], camera_aim[2]), 
                      glm.vec3(camera_up[0], camera_up[1], camera_up[2]))    
    projection = glm.perspective(glm.radians(field_of_view), float(window_size[0])/float(window_size[1]), 0.01, far)
    light_pos = np.array(camera_pos, dtype=np.float32) + np.array(camera_up, dtype=np.float32) * 10

    # Uniforms de transformação para o vertex shader
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    # Uniforms de iluminação para o fragment shader
    glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 1.0, 1.0, 1.0)
    glUniform3fv(glGetUniformLocation(shader_program, "viewPos"), 1, camera_pos)

    glUniform3f(glGetUniformLocation(shader_program, "ambientColor"), 0.3, 0.3, 0.3)
    glUniform3f(glGetUniformLocation(shader_program, "lightColor"), 1.0, 1.0, 0.5)
    glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, light_pos)
    glUniform1f(glGetUniformLocation(shader_program, "shininess"), 64.0)

    for model in scene:
        glBindVertexArray(model[-2])

        texture_id = model[3]
        use_texture = texture_id is not None
        glUniform1i(glGetUniformLocation(shader_program, "useTexture"), use_texture)

        if use_texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glUniform1i(glGetUniformLocation(shader_program, "bindTexture"), 0)

        translation = model[5]
        rotation = model[6]
        scale = model[7]

        model_matrix = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2])) * \
                    glm.rotate(glm.mat4(1.0), rotation[0], glm.vec3(rotation[1], rotation[2], rotation[3])) * \
                    glm.scale(glm.mat4(1.0), glm.vec3(scale[0], scale[1], scale[2]))
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))

        glDrawArrays(GL_TRIANGLES, 0, model[-1])

        if use_texture:
            glBindTexture(GL_TEXTURE_2D, 0)

        glBindVertexArray(0)


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
    global keys_used

    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    if action == glfw.REPEAT or action == glfw.PRESS:
        keys_used[key] = True
    elif action == glfw.RELEASE:
        keys_used[key] = False


def process_user_interaction(window):
    angle_rotate_rads = 0.01

    if keys_used.get(glfw.KEY_LEFT, False) or keys_used.get(glfw.KEY_A, False):
        rotate_view_pos_horizontal(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_RIGHT, False) or keys_used.get(glfw.KEY_D, False):
        rotate_view_pos_horizontal(angle_rotate_rads)

    if keys_used.get(glfw.KEY_UP, False) or keys_used.get(glfw.KEY_W, False):
        rotate_view_pos_vertical(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_DOWN, False) or keys_used.get(glfw.KEY_S, False):
        rotate_view_pos_vertical(angle_rotate_rads)


def main():
    glfw.init()

    window = glfw.create_window(window_size[0], window_size[1], "Scene Viewer", None, None)
    glfw.make_context_current(window) # cria um contexto opengl
    glfw.swap_interval(1)
    
    glfw.set_framebuffer_size_callback(window, my_update_window_size)
    glfw.set_key_callback(window, my_keyboard)

    glfw.maximize_window(window)

    my_init()

    while not glfw.window_should_close(window):
        glfw.poll_events() # escutando eventos
        process_user_interaction(window)

        my_render()

        glfw.swap_buffers(window) # alternando os buffers

    glfw.terminate()


if __name__ == "__main__":
    main()