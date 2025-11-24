#version 330 core

in vec3 outFragColor;
in vec2 outTexCoord;

uniform sampler2D bindTexture;

out vec4 FragColor;

void main()
{
    FragColor = texture(bindTexture, outTexCoord) * vec4(outFragColor, 1.0);
}