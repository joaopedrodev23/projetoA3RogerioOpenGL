#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 viewPos;
uniform vec3 lightPos;

out vec3 outNormal;
out vec3 outFragPos;

void main (void)
{
    outNormal = mat3(transpose(inverse(model))) * aNormal;
    outFragPos = vec3(model * vec4(aPos, 1.0));
 
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
