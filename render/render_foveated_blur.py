import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from render_blur import GaussianBlurRenderer, _create_rgb8_texture, _create_fbo_with_color_tex, _load_text, FULLSCREEN_VERT

class FoveatedBlurRenderer:
    """Foveated blur renderer that applies blur only in the periphery.
    
    This renderer:
    1. Keeps the original (sharp) frame
    2. Applies full Gaussian blur using GaussianBlurRenderer
    3. Composites sharp and blurred frames based on distance from gaze position
    
    Usage:
        r = FoveatedBlurRenderer(w, h, blur_glsl_path="shader/blur.glsl", 
                                  composite_glsl_path="shader/foveal_composite.glsl")
        r.set_blur_params(radius_rgb=(0,2,6), sigma_rgb=(0.001,1.0,3.0))
        r.set_foveal_params(foveal_radius=0.05, transition_width=0.1)
        out_tex = r.process(frame_np, gaze_pos=(0.5, 0.5))  # gaze_pos in normalized coords [0,1]
    """
    
    def __init__(self, width: int, height: int, blur_glsl_path: str, 
                 composite_glsl_path: str, input_format: int = GL_RGB):
        self.w = int(width)
        self.h = int(height)
        self.input_format = input_format
        
        #Initialize the Gaussian blur renderer for the blur pipeline
        self.blur_renderer = GaussianBlurRenderer(width, height, blur_glsl_path, input_format)
        
        #Create texture for the original (unblurred) frame
        #We need to keep this separate from the blur pipeline
        self.tex_original = _create_rgb8_texture(self.w, self.h)
        
        # Create texture for the final composited output
        self.tex_composite_out = _create_rgb8_texture(self.w, self.h)
        
        # Create FBO for compositing pass
        self.fbo_composite = _create_fbo_with_color_tex(self.tex_composite_out)
        
        # Load and compile the foveated compositing shader
        if not os.path.exists(composite_glsl_path):
            raise FileNotFoundError(f"Composite shader not found: {composite_glsl_path}")
        
        composite_frag = _load_text(composite_glsl_path)
        self.composite_prog = compileProgram(
            compileShader(FULLSCREEN_VERT, GL_VERTEX_SHADER),
            compileShader(composite_frag, GL_FRAGMENT_SHADER)
        )
        
        # Cache uniform locations for compositing shader
        self._cache_composite_uniforms()
        
        # Create VAO for fullscreen triangle
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindVertexArray(0)
        
        # Default foveal parameters (in normalized screen coordinates)
        self.foveal_radius = 0.05  # 5% of screen
        self.transition_width = 0.1  # 10% transition zone
        self.aspect_ratio = 1.0   # corrected via set_foveal_params
        self.blur_active = True
        
        #Fallback gaze pos
        self.gaze_pos = (0.5, 0.5)

        #Luminance correction strength (0 off, 1.0 full correction)
        self.lum_correction = 0.0
    
    def _cache_composite_uniforms(self):
        """Cache uniform locations for the compositing shader."""
        glUseProgram(self.composite_prog)
        
        self.loc_original = glGetUniformLocation(self.composite_prog, "uOriginal")
        self.loc_blurred = glGetUniformLocation(self.composite_prog, "uBlurred")
        self.loc_gaze_pos = glGetUniformLocation(self.composite_prog, "uGazePos")
        self.loc_foveal_radius = glGetUniformLocation(self.composite_prog, "uFovealRadius")
        self.loc_transition_width = glGetUniformLocation(self.composite_prog, "uTransitionWidth")
        self.loc_aspect_ratio = glGetUniformLocation(self.composite_prog, "uAspectRatio")
        self.loc_lum_correction = glGetUniformLocation(self.composite_prog, "uLumCorrection")
        
        #Bind texture samplers to texture units (do this once)
        glUniform1i(self.loc_original, 0)  # texture unit 0
        glUniform1i(self.loc_blurred, 1)   # texture unit 1
        
        glUseProgram(0)
    
    def set_blur_params(self, radius_rgb=(0, 2, 6), sigma_rgb=(0.001, 1.0, 3.0)):
        """Set blur parameters (passed through to GaussianBlurRenderer)."""
        self.blur_renderer.set_params(radius_rgb=radius_rgb, sigma_rgb=sigma_rgb)
    
    def set_foveal_params(self, foveal_radius=0.05, transition_width=0.1, aspect_ratio=1.0):
        """Set foveal region parameters.
        
        Args:
            foveal_radius: Radius of sharp foveal region in normalized coordinates [0,1]
            transition_width: Width of transition zone from sharp to full blur
            aspect_ratio: Screen width/height ratio: corrects the foveal boundary to be circular
        """
        self.foveal_radius = float(foveal_radius)
        self.transition_width = float(transition_width)
        self.aspect_ratio = float(aspect_ratio)
    
    def set_blur_active(self, active: bool):
        """Enable or disable blur. When False, process() returns the unmodified input frame."""
        self.blur_active = bool(active)

    def set_lum_correction(self, strength: float):
        """Set luminance correction strength for the blurred region.

        The differential per-channel blur radii cause bright pixels to appear
        yellow in the periphery (blue energy spreads more than red/green).
        This correction rescales the blurred pixel's RGB to match the sharp
        pixel's perceived luminance, smoothing out the yellow cast.

        Args:
            strength: 0.0 = no correction (full yellow cast),
                      1.0 = fully luminance-preserving.
        """
        self.lum_correction = float(max(0.0, min(1.0, strength)))

    def upload_original_frame(self, frame_rgb: np.ndarray):
        """Upload the original frame to tex_original."""
        if frame_rgb is None:
            raise ValueError("frame_rgb is None")
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] not in (3, 4):
            raise ValueError(f"Expected uint8 HxWx3/4, got dtype={frame_rgb.dtype}, shape={frame_rgb.shape}")
        if frame_rgb.shape[0] != self.h or frame_rgb.shape[1] != self.w:
            raise ValueError(f"Frame size {frame_rgb.shape[1]}x{frame_rgb.shape[0]} != renderer {self.w}x{self.h}")
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        
        glBindTexture(GL_TEXTURE_2D, self.tex_original)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, 
                        self.input_format, GL_UNSIGNED_BYTE, frame_rgb)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def _draw_fullscreen(self):
        """Draw a fullscreen triangle."""
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
    
    def process(self, frame_rgb: np.ndarray, gaze_pos=(0.5, 0.5)) -> int:
        """Full foveated blur pipeline.
        
        Args:
            frame_rgb: Input frame as numpy array (H, W, 3) uint8
            gaze_pos: Gaze position as (x, y) tuple in normalized coordinates [0,1]
                     (0,0) = top-left, (1,1) = bottom-right
        
        Returns:
            GL texture ID of the composited output (sharp in fovea, blurred in periphery)
        """
        self.gaze_pos = tuple(gaze_pos)
        
        # 1) Upload original frame (for foveal region)
        self.upload_original_frame(frame_rgb)

        if not self.blur_active:
            return self.tex_original

        # 2) Process with Gaussian blur renderer (full blur)
        #    This handles uploading to its own tex_in and returns the blurred output
        tex_blurred = self.blur_renderer.process(frame_rgb)
        
        # 3) Composite pass: blend original and blurred based on eccentricity
        # Save previous viewport
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        glViewport(0, 0, self.w, self.h)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_composite)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.composite_prog)
        
        # Bind original texture to unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_original)
        
        # Bind blurred texture to unit 1
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex_blurred)
        
        # Set uniforms
        glUniform2f(self.loc_gaze_pos, self.gaze_pos[0], self.gaze_pos[1])
        glUniform1f(self.loc_foveal_radius, self.foveal_radius)
        glUniform1f(self.loc_transition_width, self.transition_width)
        glUniform1f(self.loc_aspect_ratio, self.aspect_ratio)
        glUniform1f(self.loc_lum_correction, self.lum_correction)
        
        # Draw fullscreen triangle
        self._draw_fullscreen()
        
        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Restore viewport
        try:
            glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])
        except Exception:
            pass
        
        return self.tex_composite_out
    
    def cleanup(self):
        """Cleanup all OpenGL resources."""
        # Cleanup the blur renderer
        self.blur_renderer.cleanup()
        
        # Delete our own resources
        glDeleteProgram(self.composite_prog)
        glDeleteTextures([self.tex_original, self.tex_composite_out])
        glDeleteFramebuffers(1, [self.fbo_composite])
        glDeleteVertexArrays(1, [self.vao])
