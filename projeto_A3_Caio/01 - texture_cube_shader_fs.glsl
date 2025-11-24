#version 330 core

in vec3 outFragColor;
in vec2 outTexCoord;

uniform bool useTexture;
uniform sampler2D bindTexture;

out vec4 FragColor;

void main()
{
    if (useTexture)
        FragColor = texture(bindTexture, outTexCoord);
    else
        FragColor = vec4(outFragColor, 1.0);
}