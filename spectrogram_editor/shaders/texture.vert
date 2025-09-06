# Basic Vertex Shader for Texture Rendering
# Used with fragment shaders for spectrogram display

#version 460 core

// Input vertex attributes
in vec2 aPosition;   // Vertex position (-1 to 1)
in vec2 aTexCoord;   // Texture coordinates (0 to 1)

// Output to fragment shader
out vec2 uv;

// Uniforms
uniform mat4 uMVPMatrix;  // Model-View-Projection matrix

void main() {
    // Pass texture coordinates to fragment shader
    uv = aTexCoord;
    
    // Transform vertex position
    gl_Position = uMVPMatrix * vec4(aPosition, 0.0, 1.0);
}