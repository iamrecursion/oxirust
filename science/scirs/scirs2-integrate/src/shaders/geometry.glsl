#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vertexColor[];
out vec3 geoColor;

void main() {
    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        geoColor = vertexColor[i];
        EmitVertex();
    }
    EndPrimitive();
}