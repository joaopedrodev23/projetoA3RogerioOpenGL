#version 330 core

in vec3 outNormal;
in vec3 outFragPos;

uniform vec3 objectColor;
uniform vec3 viewPos;

uniform vec3 lightPos;
uniform vec3 lightColor;

uniform vec3 ambientColor;
uniform float shininess;

out vec4 FragColor;

void main (void)
{
    vec3 normal = normalize(outNormal);
    vec3 lightDir = normalize(lightPos - outFragPos);
    vec3 viewDir = normalize(viewPos - outFragPos);
    
    vec3 fragColor = ambientColor * objectColor;

    // Difusa (Lambert)
    float cos_teta = max(0.0, dot(normal, lightDir));
    vec3 diffuseColor = cos_teta * lightColor;
    fragColor += diffuseColor * objectColor;

    // Especular (Phong)
    if (cos_teta != 0)
    {
        vec3 reflectDir = normalize(reflect(-lightDir, normal));
        float cos_phi = max(0.0, dot(viewDir, reflectDir));
        vec3 specularColor = pow(cos_phi, shininess) * lightColor;

        fragColor += specularColor * objectColor;
    }

    FragColor = vec4(fragColor, 1.0);
}

