#version 330 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float time;
uniform int effectType;

void main() {
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    // Apply different effects based on effectType
    if (effectType == 1) {
        // Grayscale effect
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = vec3(gray);
    } else if (effectType == 2) {
        // Sepia effect
        color = vec3(
            dot(color, vec3(0.393, 0.769, 0.189)),
            dot(color, vec3(0.349, 0.686, 0.168)),
            dot(color, vec3(0.272, 0.534, 0.131))
        );
    }
    
    FragColor = vec4(color, 1.0);
}