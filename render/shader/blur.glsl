#version 330 core

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uInput;

//Texel size
uniform vec2 uTexelSize;

//Per channel blur radii (R=0, G=2, B=6) from Schaeffel
uniform ivec3 uRadiusRGB;

//Precomputed normalized Gaussian weights per channel, indices 0..MAX_RADIUS
//Uploaded from CPU when params change; avoids per-fragment exp() calls
uniform float uWeightsR[11];
uniform float uWeightsG[11];
uniform float uWeightsB[11];

//Compile time direction, horizontal is vec2(1,0), vertical is vec2(0,1)
#ifndef BLUR_DIR
  #error "BLUR_DIR not defined. Define BLUR_DIR as vec2(1,0) or vec2(0,1) before compiling"
#endif

//Max blur radius, any larger and the sampling size becomes too large
const int MAX_RADIUS = 10;

//Sample RGB values of the pixel
vec3 sampleRGB(vec2 uv){
  return texture(uInput, uv).rgb;
}





//Blur logic
//-----------------

//perform 1D separable gaussian blur in the BLUR_DIR direction
//uses precomputed per-channel weight arrays; no exp() per fragment
vec3 blurSeparableGaussian(vec2 uv){
  //clamp to avoid unsafe values
  int radiusR = clamp(uRadiusRGB.r, 0, MAX_RADIUS);
  int radiusG = clamp(uRadiusRGB.g, 0, MAX_RADIUS);
  int radiusB = clamp(uRadiusRGB.b, 0, MAX_RADIUS);

  //accumulator for the three colour values
  vec3 accumulator = vec3(0.0);

  //Center tap
  {
    vec3 c = sampleRGB(uv);
    accumulator += vec3(c.r * uWeightsR[0], c.g * uWeightsG[0], c.b * uWeightsB[0]);
  }

  //Symmetric taps in 1D to either side
  for (int i = 1; i <= MAX_RADIUS; i++){
    //if i exceeds radii for all channels, stop sampling
    if (i > radiusR && i > radiusG && i > radiusB) break;

    //Which pixels are we going to?
    vec2 offset = float(i) * (uTexelSize * BLUR_DIR);

    //Sample the points
    vec3 c1 = sampleRGB(uv + offset);
    vec3 c2 = sampleRGB(uv - offset);

    //Apply precomputed weight, 0 if outside channel radius
    float wR = (i <= radiusR) ? uWeightsR[i] : 0.0;
    float wG = (i <= radiusG) ? uWeightsG[i] : 0.0;
    float wB = (i <= radiusB) ? uWeightsB[i] : 0.0;

    accumulator += vec3((c1.r + c2.r) * wR,
                        (c1.g + c2.g) * wG,
                        (c1.b + c2.b) * wB);
  }

  return accumulator;
}



void main(){
  vec3 outRGB = blurSeparableGaussian(vUV);
  FragColor = vec4(outRGB, 1.0);
}
