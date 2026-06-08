//A fragment shader applying chromatic filtering

#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uFrame;

// Gaze&frame info
uniform vec2  uResolution;   // (width, height) in pixels
uniform vec2  uGazePx;        // gaze in pixels (same coordinate space as frame)
uniform int   uMaskMode;      // 0=full-frame, 1=circle hard, 2=gaussian feather

//Mask parameters
uniform float uRadiusPx;      // foveal radius (no filter) in pixels
uniform float uFeatherPx;     // transition width in pixels (for gaussian-ish edge)

// Filter parameter
uniform float uFilterStrength; // 0..1

//Distance from this fragment to gaze in pixels
float gazeDistancePx(vec2 fragUV) {
    vec2 fragPx = fragUV * uResolution;
    return length(fragPx - uGazePx);
}

// Mask weight: 0 near gaze, 1 in periphery
float peripheralWeight(float distPx) {
    if (uMaskMode == 0) {
        return 1.0; // full-frame filtering
    }

    // Hard circle: step outside radius
    if (uMaskMode == 1) {
        return step(uRadiusPx, distPx); //returns 0 inside gaze circle, returns 1 outside
    }

    //Gaussian feather
    //Inside radius = 0
    //Outside rises smoothly towards 1
    float x = max(distPx - uRadiusPx, 0.0); //get the distance of frag and gaze circle

    //no div by 0
    float feather = max(uFeatherPx, 1e-5);
    float sigma = feather / 3.0;
    float t = x/sigma;
    float w = 1.0 - exp(-0.5 * t * t);

    return clamp(w,0.0,1.0)
}






void main() {

    float distPx = gazeDistancePx(vUV);
    float w = peripheralWeight(distPx) * clamp(uFilterStrength, 0.0, 1.0);

    //Get the original and filtered colour and mix them with weight w
    vec3 original = texture(uFrame, vUV).rgb;
    vec3 filtered = texture(uFiltered,vUV).rgb;
    vec3 outRgb = mix(original, filtered, w);

    //Output new fragcolor
    FragColor = vec4(outRgb, 1.0);
}
