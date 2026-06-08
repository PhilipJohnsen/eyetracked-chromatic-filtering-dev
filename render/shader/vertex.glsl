#version 330 core

// UV coordinates pass to frag shader

out vec2 vUV;

void main(){
  //Fullscreen triangle using vertex ID
  vec2 pos;
  if (gl_VertexID==0) pos = vec2 (-1.0, -1.0);
  if (gl_VertexID==1) pos = vec2(3.0,-1.0);
  if (gl_VertexID==2) pos = vec2 (-1.0,3.0);

  gl_Position = vec4(pos, 0.0, 1.0);

  //Map from clip space (-1 to 1) to UV space (0 to 1)
  vUV = 0.5 * (pos+1.0);
}
