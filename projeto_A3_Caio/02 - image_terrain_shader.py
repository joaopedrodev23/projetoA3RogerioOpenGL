import glfw
import glm

from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as gls

from geometry_utils import *

from PIL import Image


field_of_view = 60
window_size = [600, 600]

camera_pos = np.array([0, 0, 10], dtype=np.float32)
camera_aim = np.array([0, 0, 0], dtype=np.float32)
camera_up = np.array([0, 0, 1], dtype=np.float32)
far = 1000.0

shader_gouraud_shading_program = None
shader_phong_shading_program = None

scene = None
terrain_heightmap = None
terrain_bounding_box = None

current_shading_mode = 0

keys_used = {}
WALKER_HEIGHT_OFFSET = 1.6

HM_SCALE_WORLD = 300.0     # escala no espaço do mundo (tamanho do terreno)
HM_HEIGHT_SCALE = 5.0      # amplitude do terreno


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


def create_vao(model_vertices_pos, model_vertices_normals):
    vao_id = glGenVertexArrays(1)
    glBindVertexArray(vao_id)

    create_vbo(GL_ARRAY_BUFFER, model_vertices_pos, 0)
    create_vbo(GL_ARRAY_BUFFER, model_vertices_normals, 1)

    glBindVertexArray(0)

    return vao_id


# ################################################################################################
# OpenGL - Funções auxiliares
# ################################################################################################

def get_heightmap_height(heightmap, heightmap_bounding_box, x_world, z_world):
    x_proportion = (x_world - heightmap_bounding_box[0][0]) / (heightmap_bounding_box[1][0] - heightmap_bounding_box[0][0])
    z_proportion = (z_world - heightmap_bounding_box[0][2]) / (heightmap_bounding_box[1][2] - heightmap_bounding_box[0][2])

    grid_height, grid_width = heightmap.shape
    x = grid_width * x_proportion
    z = grid_height * z_proportion

    # 1. Obter as dimensões do heightmap para verificação de limites
    max_z, max_x = heightmap.shape
    max_z -= 1
    max_x -= 1

    # 2. Verificar e limitar as coordenadas (clamp)
    # Se o ponto estiver fora do mapa, ele retorna a altura da borda
    x = np.clip(x, 0, max_x)
    z = np.clip(z, 0, max_z)

    # 3. Encontrar os índices inteiros (floor)
    x1 = int(np.floor(x))
    z1 = int(np.floor(z))

    # x2 e z2 são os índices para o próximo ponto (o teto, que é o floor + 1)
    # Garante que não ultrapasse os limites do array
    x2 = min(x1 + 1, max_x)
    z2 = min(z1 + 1, max_z)

    # Se o ponto cair exatamente na borda, a interpolação não faz sentido 
    # e podemos retornar a altura do ponto x1, z1.
    if x1 == x2 and z1 == z2:
        return heightmap[z1, x1]

    # 4. Calcular os pesos (frações)
    # np.fmod retorna o resto da divisão, que é a parte fracionária
    delta_x = x - x1
    delta_z = z - z1
    
    # 5. Obter as Alturas Vizinhas
    # Usando o clamp (min/max) para pegar os 4 cantos mais próximos
    h00 = heightmap[z1, x1] # Canto superior-esquerdo
    h10 = heightmap[z1, x2] # Canto superior-direito
    h01 = heightmap[z2, x1] # Canto inferior-esquerdo
    h11 = heightmap[z2, x2] # Canto inferior-direito

    # 6. Interpolação Linear em X
    
    # Interpola na linha superior (z1)
    # (h00 * (1 - delta_x)) + (h10 * delta_x)
    h_top = h00 * (1.0 - delta_x) + h10 * delta_x
    
    # Interpola na linha inferior (z2)
    # (h01 * (1 - delta_x)) + (h11 * delta_x)
    h_bottom = h01 * (1.0 - delta_x) + h11 * delta_x

    # 7. Interpolação Linear Final em Z
    
    # Interpola entre as duas alturas resultantes (h_top e h_bottom)
    # (h_top * (1 - delta_z)) + (h_bottom * delta_z)
    final_height = h_top * (1.0 - delta_z) + h_bottom * delta_z

    return final_height


def create_heightmap(heightmap_image_path, scale_world, height_scale):
    heightmap_image = Image.open(heightmap_image_path).convert('L')  # grayscale
    heightmap = np.asarray(heightmap_image, dtype=np.float32) / 255.0

    grid_height, grid_width = heightmap.shape

    # grid spacing
    sx = scale_world / (grid_width - 1)
    sz = scale_world / (grid_height - 1)
    half_w = scale_world * 0.5
    half_h = scale_world * 0.5
    
    # heightmap = matriz 2D numpy (float32)
    heightmap = heightmap * height_scale  # escala em paralelo (C interno)

    # cria grids de coordenadas
    x = np.arange(grid_width) * sx - half_w
    z = np.arange(grid_height) * sz - half_h
    X, Z = np.meshgrid(x, z)

    # empilha tudo em um array 3D (h*w, 3)
    vertices_pos_heightmap = np.stack([X, heightmap, Z], axis=-1).reshape(-1, 3)
   
    # Criar matriz de índices base
    grid_indices = np.arange(grid_height * grid_width, dtype=np.uint32).reshape(grid_height, grid_width)

    # Pegar os quatro vértices de cada quadrado (a,b,c,d)
    a = grid_indices[:-1, :-1].ravel()
    b = grid_indices[:-1, 1:].ravel()
    c = grid_indices[1:, :-1].ravel()
    d = grid_indices[1:, 1:].ravel()

    # Construir triângulos (duas faces por quadrado)
    # Triângulo 1: a, c, b
    # Triângulo 2: b, c, d
    indices = np.column_stack([
        np.stack([a, c, b], axis=1),
        np.stack([b, c, d], axis=1)
    ]).ravel()

    # Construir faces_heightmap (opcional)
    faces_heightmap = np.concatenate([
        np.stack([a, c, b], axis=1),
        np.stack([b, c, d], axis=1)
    ], axis=0).reshape(-1, 3, 1).astype(np.uint32)

    # Reorganiza os índices em grupos de 3 (um triângulo por linha)
    triangles = indices.reshape(-1, 3)

    # Pega as posições dos 3 vértices de cada triângulo
    a = vertices_pos_heightmap[triangles[:, 0]]
    b = vertices_pos_heightmap[triangles[:, 1]]
    c = vertices_pos_heightmap[triangles[:, 2]]

    # Calcula as normais das faces (vetorial)
    normals = np.cross(b - a, c - a)

    # Normaliza todas as normais de uma vez
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-9)

    # Soma as normais de cada face nos vértices correspondentes
    vertices_normals_heightmap = np.zeros_like(vertices_pos_heightmap)
    np.add.at(vertices_normals_heightmap, triangles[:, 0], normals)
    np.add.at(vertices_normals_heightmap, triangles[:, 1], normals)
    np.add.at(vertices_normals_heightmap, triangles[:, 2], normals)

    # Normaliza novamente por vértice
    norms = np.linalg.norm(vertices_normals_heightmap, axis=1, keepdims=True)
    vertices_normals_heightmap = vertices_normals_heightmap / (norms + 1e-9)

    faces_normals_heightmap = compute_faces_normals(vertices_pos_heightmap, faces_heightmap)

    return heightmap, vertices_pos_heightmap, vertices_normals_heightmap, faces_heightmap, faces_normals_heightmap


def create_scene():
    heightmap, vertices_pos_heightmap, vertices_normals_heightmap, faces_heightmap, faces_normals_heightmap = create_heightmap('Heightmaps/heightmap_01.png', HM_SCALE_WORLD, HM_HEIGHT_SCALE)
    heightmap_bounding_box = compute_bounding_box(vertices_pos_heightmap, faces_heightmap)

    # scene = [[vertices_pos, vertices_normals, faces, faces_normals, color, translation, rotation, scale],...]
    scene = [[vertices_pos_heightmap, vertices_normals_heightmap, faces_heightmap, faces_normals_heightmap, [0.4941, 0.5491, 0.3294], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1, 1, 1]]]

    return scene, heightmap, heightmap_bounding_box


def compute_camera_center_terrain(heightmap, heightmap_bounding_box):
    center = get_bounding_box_center(heightmap_bounding_box)
    center_height = get_heightmap_height(heightmap, heightmap_bounding_box, center[0], center[2])

    return np.array([center[0], center_height + WALKER_HEIGHT_OFFSET, center[2]], dtype=np.float32), \
           np.array([center[0], center_height + WALKER_HEIGHT_OFFSET, center[2] - 1], dtype=np.float32), \
           np.array([0, 1, 0], dtype=np.float32), HM_SCALE_WORLD


def compute_scene_bounding_box(scene):
    if len(scene) == 0:
        return [[0, 0, 0], [0, 0, 0]]
    
    first_model = scene[0]
    scene_bounding_box = translate_bounding_box(compute_bounding_box(first_model[0], first_model[2]), first_model[5])

    for i in range(1, len(scene)):
        model = scene[i]
        model_bounding_box = translate_bounding_box(compute_bounding_box(model[0], model[2]), model[5])
        scene_bounding_box = union_bounding_boxes(scene_bounding_box, model_bounding_box)

    return scene_bounding_box


def get_vertices_lists_flat_shading(vertices, faces, faces_normals):
    vertex_pos_list = []
    vertex_normals_list = []

    for i in range(len(faces)):
        face = faces[i]
        face_normal = faces_normals[i]

        for j in range(3):
            vertex_index = face[j][0]
            vertex = vertices[vertex_index]
            vertex_pos_list.append(vertex)
            vertex_normals_list.append(face_normal)

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32)


def get_vertices_lists_vertex_shading(vertices, faces, vertices_normals):
    vertex_pos_list = []
    vertex_normals_list = []

    for i in range(len(faces)):
        face = faces[i]

        for j in range(3):
            vertex_index = face[j][0]
            
            vertex = vertices[vertex_index]
            vertex_normal = vertices_normals[vertex_index]

            vertex_pos_list.append(vertex)
            vertex_normals_list.append(vertex_normal)

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32)


def my_init():
    glClearColor(135.0/255.0, 206.0/255.0, 235.0/255.0, 1)
    glEnable(GL_DEPTH_TEST)
    
    # carrega o modelo
    global scene, terrain_heightmap, terrain_bounding_box
    scene, terrain_heightmap, terrain_bounding_box = create_scene()

    # calcula posição inicial da camera
    global camera_pos, camera_aim, camera_up, far
    camera_pos, camera_aim, camera_up, far = compute_camera_center_terrain(terrain_heightmap, terrain_bounding_box)

    # compila os programas shaders:
    # 1 - um que implementa o shading de Gouraud, implementado pelo OpenGL Legado
    # 2 - outro que implementa o shading de Phong
    global shader_gouraud_shading_program
    shader_gouraud_shading_program = create_shader_program('02 - image_terrain_shader_gouraud_vs.glsl', '02 - image_terrain_shader_gouraud_fs.glsl')

    global shader_phong_shading_program
    shader_phong_shading_program = create_shader_program('02 - image_terrain_shader_phong_vs.glsl', '02 - image_terrain_shader_phong_fs.glsl')

    # Para cada model da cena cria dois VAOs: 
    # 1 - um para flat shading com a mesma normal para os 3 vértices de uma mesma face e
    # 2 - outro para gouraud e phong shading com a normal calculada por vértice como sendo a média das normais das faces em torno do vértice
    for model in scene:
        model_vertices_pos, model_vertices_normals = get_vertices_lists_flat_shading(model[0], model[2], model[3])
        vao_flat_shading_id = create_vao(model_vertices_pos, model_vertices_normals)

        model_vertices_pos, model_vertices_normals = get_vertices_lists_vertex_shading(model[0], model[2], model[1])
        vao_vertex_shading_id = create_vao(model_vertices_pos, model_vertices_normals)

        model.append(vao_flat_shading_id)
        model.append(vao_vertex_shading_id)
        model.append(len(model_vertices_pos))


def get_current_vao_index():
    if current_shading_mode % 3 == 0:
        return 8
    
    if current_shading_mode % 3 == 1:
        return 9

    return 9


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
    view = glm.lookAt(glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2]), 
                      glm.vec3(camera_aim[0], camera_aim[1], camera_aim[2]), 
                      glm.vec3(camera_up[0], camera_up[1], camera_up[2]))    
    projection = glm.perspective(glm.radians(field_of_view), float(window_size[0])/float(window_size[1]), 0.01, far)
    light_pos = camera_pos + camera_up * 10

    # Uniforms de transformação para o vertex shader
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    # Uniforms de iluminação para o fragment shader
    glUniform3fv(glGetUniformLocation(current_shader_program, "viewPos"), 1, camera_pos)

    glUniform3f(glGetUniformLocation(current_shader_program, "ambientColor"), 0.25, 0.25, 0.25)
    glUniform3f(glGetUniformLocation(current_shader_program, "lightColor"), 1.0, 1.0, 0.5)
    glUniform3fv(glGetUniformLocation(current_shader_program, "lightPos"), 1, light_pos)
    glUniform1f(glGetUniformLocation(current_shader_program, "shininess"), 64.0)

    for model in scene:
        current_vao_index = get_current_vao_index()
        glBindVertexArray(model[current_vao_index])

        translation = model[5]
        rotation = model[6]
        scale = model[7]

        model_matrix = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2])) * \
                    glm.rotate(glm.mat4(1.0), rotation[0], glm.vec3(rotation[1], rotation[2], rotation[3])) * \
                    glm.scale(glm.mat4(1.0), glm.vec3(scale[0], scale[1], scale[2]))
        glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))

        objectColor = np.array(model[4])
        glUniform3fv(glGetUniformLocation(current_shader_program, "objectColor"), 1, objectColor)

        glDrawArrays(GL_TRIANGLES, 0, model[-1])
    
        glBindVertexArray(0)
    
    glUseProgram(0)


# ################################################################################################
# GLFW - Funções auxiliares
# ################################################################################################

def my_update_window_size(window, width, height):
    global window_size
    window_size = [width, height]


def walk_forward(distance):
    global camera_pos, camera_aim

    view_direction = camera_aim - camera_pos
    view_direction = view_direction / np.linalg.norm(view_direction)

    camera_pos = camera_pos + view_direction * distance
    camera_pos[1] = get_heightmap_height(terrain_heightmap, terrain_bounding_box, camera_pos[0], camera_pos[2]) + WALKER_HEIGHT_OFFSET

    camera_aim = camera_pos + view_direction


def rotate_walker(angle_rotate_rads):
    global camera_aim

    view_direction = camera_aim - camera_pos
    side_direction = np.cross(view_direction, np.array([0.0, 1.0, 0.0]))
    up = np.cross(side_direction, view_direction)
    up = up / np.linalg.norm(up)

    rotate_matrix = get_rotate_matrix(angle_rotate_rads, up)

    new_view_direction = np.dot(view_direction, rotate_matrix.T)
    camera_aim = camera_pos + new_view_direction


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
    angle_rotate_rads = 0.01
    step_walk = 0.03

    if keys_used.get(glfw.KEY_LEFT, False) or keys_used.get(glfw.KEY_A, False):
        rotate_walker(angle_rotate_rads)

    if keys_used.get(glfw.KEY_RIGHT, False) or keys_used.get(glfw.KEY_D, False):
        rotate_walker(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_UP, False) or keys_used.get(glfw.KEY_W, False):
        walk_forward(step_walk)

    if keys_used.get(glfw.KEY_DOWN, False) or keys_used.get(glfw.KEY_S, False):
        walk_forward(-step_walk)


# ################################################################################################
# Main - Programa Principal
# ################################################################################################

def main():

    glfw.init()

    window = glfw.create_window(window_size[0], window_size[1], "Terreno", None, None)
    glfw.make_context_current(window) 
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