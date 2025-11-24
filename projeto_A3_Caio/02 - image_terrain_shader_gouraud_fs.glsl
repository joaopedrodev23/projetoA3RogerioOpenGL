#version 330 core

in vec3 outFragColor;

out vec4 FragColor;

void main()
{
    FragColor = vec4(outFragColor, 1.0);
}