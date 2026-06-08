#version 330 core

in vec2 vUV;
out vec4 FragColor;

//Input textures
uniform sampler2D uOriginal;  // sharp, unblurred frame
uniform sampler2D uBlurred;   // fully blurred frame

//Gaze position in NDC [0,1] x [0,1]
uniform vec2 uGazePos;

//Foveal region param (in normalized screen coordinates)
//uFovealRadius: radius where image is completely sharp (no blur)
//uTransitionWidth: width of transition zone from sharp to full blur
uniform float uFovealRadius;
uniform float uTransitionWidth;
//Aspect ratio (w/h): scales x so the foveal boundary is
//circular in physical screen space rather than elliptical in UV space because UV space isnt linear
uniform float uAspectRatio;

// Luminance correction strength [0.0, 1.0].
// 0.0 = no correction (full yellow cast), 1.0 = fully luminance-preserving.
uniform float uLumCorrection;

void main() {
    //Calculate distance from current pixel to gaze position.
    vec2 diff = vUV - uGazePos;
    diff.x *= uAspectRatio;
    float distance = length(diff);
    
    // Calculate blend factor: 0.0 = sharp (fovea), 1.0 = blurred (periphery)
    float innerRadius = uFovealRadius;
    float outerRadius = uFovealRadius + uTransitionWidth;
    float blendFactor = smoothstep(innerRadius, outerRadius, distance);
    
    //Sample both textures
    vec3 sharp = texture(uOriginal, vUV).rgb;
    vec3 blurred = texture(uBlurred, vUV).rgb;

    // Luminance-preserving correction for the blurred region.
    const vec3 lum_weights = vec3(0.2126, 0.7152, 0.0722);
    float lum_sharp = dot(sharp,   lum_weights);
    float lum_blur  = dot(blurred, lum_weights);
    vec3 blurred_corrected = (lum_blur > 0.001)
        ? clamp(blurred * (lum_sharp / lum_blur), 0.0, 1.0)
        : blurred;
    blurred = mix(blurred, blurred_corrected, uLumCorrection);

    //Blend between sharp and blurred
    vec3 result = mix(sharp, blurred, blendFactor);
    
    FragColor = vec4(result, 1.0);
}
