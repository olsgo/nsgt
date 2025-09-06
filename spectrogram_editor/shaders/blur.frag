# Fragment Shader for Gaussian Blur Effect
# Used for real-time blur brush preview in Flutter

#version 460 core

precision mediump float;

// Input from vertex shader
in vec2 uv;

// Uniforms
uniform sampler2D uTexture;     // Input spectrogram texture
uniform vec2 uResolution;       // Texture resolution
uniform vec2 uBrushCenter;      // Brush center (0-1 coordinates)
uniform float uBrushRadius;     // Brush radius in pixels
uniform float uBlurRadius;      // Blur kernel radius
uniform float uStrength;        // Effect strength (0-1)

// Output color
out vec4 fragColor;

// Gaussian blur function
vec4 gaussianBlur(sampler2D tex, vec2 coord, vec2 resolution, float radius) {
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    
    // Calculate texel size
    vec2 texelSize = 1.0 / resolution;
    
    // Gaussian kernel parameters
    float sigma = radius / 3.0;
    float twoSigmaSquared = 2.0 * sigma * sigma;
    
    // Sample in a square around the center
    int kernelSize = int(ceil(radius * 2.0));
    
    for (int x = -kernelSize; x <= kernelSize; x++) {
        for (int y = -kernelSize; y <= kernelSize; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec2 sampleCoord = coord + offset;
            
            // Check if sample is within texture bounds
            if (sampleCoord.x >= 0.0 && sampleCoord.x <= 1.0 && 
                sampleCoord.y >= 0.0 && sampleCoord.y <= 1.0) {
                
                // Calculate Gaussian weight
                float distance = length(vec2(float(x), float(y)));
                float weight = exp(-distance * distance / twoSigmaSquared);
                
                color += texture(tex, sampleCoord) * weight;
                totalWeight += weight;
            }
        }
    }
    
    if (totalWeight > 0.0) {
        color /= totalWeight;
    }
    
    return color;
}

// Smooth brush falloff function
float brushFalloff(vec2 coord, vec2 center, float radius) {
    float distance = length(coord - center);
    float normalizedDist = clamp(distance / radius, 0.0, 1.0);
    
    // Smooth step for soft falloff
    float falloff = 1.0 - normalizedDist;
    return falloff * falloff * (3.0 - 2.0 * falloff); // Smoothstep
}

void main() {
    // Sample original texture
    vec4 original = texture(uTexture, uv);
    
    // Calculate distance from brush center
    vec2 pixelCoord = uv * uResolution;
    vec2 brushCenterPixel = uBrushCenter * uResolution;
    float brushMask = brushFalloff(pixelCoord, brushCenterPixel, uBrushRadius);
    
    if (brushMask > 0.0) {
        // Apply blur within brush area
        vec4 blurred = gaussianBlur(uTexture, uv, uResolution, uBlurRadius);
        
        // Blend based on brush mask and strength
        float effectStrength = brushMask * uStrength;
        fragColor = mix(original, blurred, effectStrength);
    } else {
        // Outside brush area, use original
        fragColor = original;
    }
}