# Fragment Shader for Displacement Warp Effect
# Used for real-time warp brush preview in Flutter

#version 460 core

precision mediump float;

// Input from vertex shader
in vec2 uv;

// Uniforms
uniform sampler2D uTexture;        // Input spectrogram texture
uniform vec2 uResolution;          // Texture resolution
uniform vec2 uBrushCenter;         // Brush center (0-1 coordinates)
uniform float uBrushRadius;        // Brush radius in pixels
uniform float uDisplacement;       // Maximum displacement in pixels
uniform vec2 uDirection;           // Displacement direction (normalized)
uniform float uFalloff;            // Falloff curve steepness
uniform float uStrength;           // Effect strength (0-1)

// Output color
out vec4 fragColor;

// Smooth brush falloff function
float brushFalloff(vec2 coord, vec2 center, float radius, float falloffCurve) {
    float distance = length(coord - center);
    float normalizedDist = clamp(distance / radius, 0.0, 1.0);
    
    // Apply falloff curve
    float falloff = 1.0 - normalizedDist;
    return pow(falloff, falloffCurve);
}

// Displacement field calculation
vec2 calculateDisplacement(vec2 coord, vec2 center, float radius, float maxDisplacement, vec2 direction, float falloffCurve) {
    float falloff = brushFalloff(coord, center, radius, falloffCurve);
    
    // Create displacement based on falloff and direction
    vec2 displacement = direction * maxDisplacement * falloff;
    
    return displacement;
}

void main() {
    // Convert UV to pixel coordinates
    vec2 pixelCoord = uv * uResolution;
    vec2 brushCenterPixel = uBrushCenter * uResolution;
    
    // Calculate displacement
    vec2 displacement = calculateDisplacement(
        pixelCoord, 
        brushCenterPixel, 
        uBrushRadius, 
        uDisplacement, 
        normalize(uDirection), 
        uFalloff
    );
    
    // Calculate brush mask for blending
    float brushMask = brushFalloff(pixelCoord, brushCenterPixel, uBrushRadius, uFalloff);
    
    if (brushMask > 0.0) {
        // Apply displacement to UV coordinates
        vec2 displacedUV = uv + (displacement / uResolution);
        
        // Sample texture at displaced coordinates
        vec4 displaced;
        if (displacedUV.x >= 0.0 && displacedUV.x <= 1.0 && 
            displacedUV.y >= 0.0 && displacedUV.y <= 1.0) {
            displaced = texture(uTexture, displacedUV);
        } else {
            // Use border color for out-of-bounds samples
            displaced = texture(uTexture, clamp(displacedUV, 0.0, 1.0));
        }
        
        // Sample original for blending
        vec4 original = texture(uTexture, uv);
        
        // Blend based on brush mask and strength
        float effectStrength = brushMask * uStrength;
        fragColor = mix(original, displaced, effectStrength);
    } else {
        // Outside brush area, use original
        fragColor = texture(uTexture, uv);
    }
}