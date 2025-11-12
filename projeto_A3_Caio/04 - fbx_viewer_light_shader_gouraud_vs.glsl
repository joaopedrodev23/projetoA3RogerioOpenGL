#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 objectColor;
uniform vec3 viewPos;

uniform vec3 ambientColor;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform float shininess;

out vec3 outFragColor;
out vec2 outTexCoord;

void main (void)
{ 
    vec3 normal = mat3(transpose(inverse(model))) * aNormal;
    
    vec3 position = vec3(model * vec4(aPos, 1));
    vec3 lightDir = normalize(lightPos - position);
    vec3 viewDir = normalize(viewPos - position);

    outFragColor = ambientColor * objectColor;

    // Difusa (Lambert)
    float cos_teta = max(0.0, dot(normal, lightDir));
    vec3 diffuseColor = cos_teta * lightColor;
    outFragColor += diffuseColor * objectColor;

    // Especular (Phong)
    if (cos_teta != 0)
    {
        vec3 reflectDir = normalize(reflect(-lightDir, normal));
        float cos_phi = max(0.0, dot(viewDir, reflectDir));
        vec3 specularColor = pow(cos_phi, shininess) * lightColor;

        outFragColor += specularColor * objectColor;
    }

    outTexCoord = aTexCoord;

    gl_Position = projection * view * model * vec4(aPos, 1.0);
}