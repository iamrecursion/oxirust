#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform float animationSpeed;

out vec3 vertexColor;

void main() {
    vec3 animatedPos = aPos;
    
    // Simple animation: wave effect
    animatedPos.y += sin(time * animationSpeed + aPos.x * 2.0) * 0.1;
    
    gl_Position = projection * view * model * vec4(animatedPos, 1.0);
    vertexColor = aColor;
}