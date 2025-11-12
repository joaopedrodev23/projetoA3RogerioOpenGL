#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 outNormal;
out vec3 outFragPos;
out vec2 outTexCoord;

void main (void)
{
    outNormal = mat3(transpose(inverse(model))) * aNormal;
    outFragPos = vec3(model * vec4(aPos, 1));
    outTexCoord = aTexCoord;

    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
