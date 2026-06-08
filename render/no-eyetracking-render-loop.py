"""
no-eyetracking-render-loop_improved.py

Goal:
  Continuous click-through overlay that renders a blurred version of the desktop
  Utilises the dxgi_capture.py implementation for high performance capture (uses DXcam import)
  Applies a gaussian blur with given kernel for each colour channel - Causes blur effect

Implementation:
  1)Create an always-on-top GLFW window that is:
        borderless
        transparent framebuffer capable
        click-through via Win32 extended styles
  2)Exclude the overlay itself from capture
     SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE).
     Correctness claim:
       - If the capture path honors the affinity flag, the captured frames
         will not include our overlay => no feedback loop.
         Per DXcam implementation using desktop duplication API from Windows, this flag should be respected
  3)render loop:
       capture_desktop_excluding_hwnd() -> blur_renderer.process(frame) -> draw overlay.
     Correctness claim:
       - We apply the filter exactly once per fresh captured frame.
"""

# -----------------------------
# DPI Awareness - Must execute BEFORE any imports that query screen dimensions
# -----------------------------
import ctypes

def _set_dpi_awareness():
    """Set DPI awareness using multiple fallback methods for compatibility."""
    #Try shcore.SetProcessDpiAwareness (Windows 8.1+)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  #PROCESS_PER_MONITOR_DPI_AWARE
        return True
    except (AttributeError, OSError):
        pass
    
    #Try user32.SetProcessDpiAwarenessContext (Windows 10 1607+)
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):  #DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return True
    except (AttributeError, OSError):
        pass
    
    #Try user32.SetProcessDPIAware (Windows Vista+)
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            return True
    except (AttributeError, OSError):
        pass
    
    return False

#Execute DPI awareness before any screen queries
if not _set_dpi_awareness():
    print("[WARNING] Failed to set DPI awareness - capture may use logical pixels instead of physical")

#---------------------
#Imports
#---------------------
import time
from pathlib import Path
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from dxgi_capture import capture_desktop_excluding_hwnd
from render_blur import GaussianBlurRenderer
from utility.load_settings import load_settings, unpack_settings


# -----------------------------
# Win32 helpers window styles. Specify transparency for clicks, keep above other windows etc.
# -----------------------------
user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
GWL_STYLE   = -16
#window styles
WS_EX_LAYERED      = 0x00080000
WS_EX_TRANSPARENT  = 0x00000020  #"hit-test transparent" => click-through
WS_EX_TOOLWINDOW   = 0x00000080  #do not show in alt-tab
WS_EX_TOPMOST      = 0x00000008  # keep above normal windows
#Normal window styles
WS_POPUP = 0x80000000
#Layered attributes
LWA_ALPHA = 0x00000002
# Display affinity:
#WDA_EXCLUDEFROMCAPTURE = 0x11 (from Win10 2004+)
WDA_EXCLUDEFROMCAPTURE = 0x00000011
#Hotkey modifiers and message
MOD_CTRL  = 0x0002
MOD_SHIFT = 0x0004
WM_HOTKEY = 0x0312

#For checking message queue for hotkey presses
class _MSG(ctypes.Structure):
    """Minimal Win32 MSG struct for PeekMessageW."""
    _fields_ = [
        ("hwnd",    ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam",  ctypes.c_size_t),
        ("lParam",  ctypes.c_ssize_t),
        ("time",    ctypes.c_uint),
        ("pt",      ctypes.c_long * 2),
    ]


def _set_clickthrough_overlay_styles(hwnd: int) -> None:
    """
    Make the GLFW window:
      - borderless (popup)
      - layered (allows transparency composition)
      - click-through (WS_EX_TRANSPARENT)
      - toolwindow (avoid taskbar/alt-tab)
      - topmost

    Correctness argument:
      * WS_EX_TRANSPARENT makes hit-testing pass through to underlying windows,
        so the user can interact with the OS "as if there was no filter active".
      * Borderless/topmost makes it behave like a full-screen overlay.
    """
    if not hwnd:
        return

    #Set popup style (borderless)
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~0x00CF0000
    style |= WS_POPUP
    user32.SetWindowLongW(hwnd, GWL_STYLE, style)
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ex |= (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)

    #Set layered window to fully opaque
    user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)

    #Force style update
    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0020 | 0x0040)  #NOSIZE|NOMOVE|FRAMECHANGED|NOACTIVATE


def _attempt_exclude_from_capture(hwnd: int) -> bool:
    """
    Exclude overlay window from desktop capture.

    Correctness claim:
      - If the capture API respects window display affinity, then no feedback
        loop will occur

    Limitations:
      - Requires Windows 10 version 2004+ for WDA_EXCLUDEFROMCAPTURE flag
      - Driver issues can prevent this from working
    """
    if not hwnd:
        return False
    ok = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    return bool(ok)


def _get_physical_monitor_resolution() -> tuple:
    """Get primary monitor resolution in physical pixels."""
    try:
        width = ctypes.windll.user32.GetSystemMetrics(0)   #SM_CXSCREEN
        height = ctypes.windll.user32.GetSystemMetrics(1)  #SM_CYSCREEN
        return (width, height)
    except Exception:
        return (None, None)


# -----------------------------
# OpenGL helpers
# -----------------------------
def _create_display_shader() -> int:
    """Minimal fullscreen triangle shader that displays a sampler2D."""
    vert = r"""
    #version 330 core
    out vec2 vUV;
    void main() {
        vec2 pos;
        if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
        if (gl_VertexID == 1) pos = vec2( 3.0, -1.0);
        if (gl_VertexID == 2) pos = vec2(-1.0,  3.0);
        gl_Position = vec4(pos, 0.0, 1.0);
        vUV = 0.5 * (pos + 1.0);
        vUV.y = 1.0 - vUV.y; // keep consistent with captured frame origin
    }
    """
    frag = r"""
    #version 330 core
    in vec2 vUV;
    out vec4 outColor;
    uniform sampler2D uTexture;
    void main() {
        outColor = texture(uTexture, vUV);
    }
    """
    return compileProgram(
        compileShader(vert, GL_VERTEX_SHADER),
        compileShader(frag, GL_FRAGMENT_SHADER),
    )


#Used for debugging capture issues, verify format/strides/contiguity of captured frames.
def _check_gl_error(tag: str) -> None:
    """Print any GL errors and clear the error flag."""
    err = glGetError()
    if err == GL_NO_ERROR:
        return
    while err != GL_NO_ERROR:
        print(f"[gl] error {tag}: 0x{err:04x}")
        err = glGetError()


# -----------------------------
#Main render-loop
# -----------------------------
def main():
    #Settings from utility/settings.txt (relative to this script)
    settings_path = Path(__file__).parent / "utility" / "settings.txt"
    settings = load_settings(str(settings_path))
    (target_fps, force_rgb, capture_format, 
     debug_gl_finish, gl_finish_interval, 
     overlay_size, overlay_pos, radius_rgb, sigma_rgb, 
     shader_path, foveal_radius, transition_width,
     gaze_source, blur_active, participant_id, session_id,
     log_gaze, log_path, lum_correction) = unpack_settings(settings)
    
    #Blur shader path (relative to this script, from settings)
    blur_glsl_path = Path(__file__).parent / shader_path

    #Query physical monitor resolution for validation
    phys_w, phys_h = _get_physical_monitor_resolution()

    #Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    #GLFW specifics:
    glfw.window_hint(glfw.DECORATED, glfw.FALSE) #borderless
    glfw.window_hint(glfw.FLOATING, glfw.TRUE) #always on top
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE) #allow alpha in framebuffer for proper compositing (even if we draw fully opaque, this is needed for correct blending with desktop)
    glfw.window_hint(glfw.FOCUSED, glfw.FALSE) #Dont steal focus from other windows, ensures no performance degradation on other windows, clickthrough should handle this perfectly
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE) #Allow resizing in case the user wants to tweak this

    #Create window context
    window = glfw.create_window(overlay_size[0], overlay_size[1], "Chromatic Filtering Overlay (no eyetracking)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)  #we select the capture pacing via settings, thus disable vsync 

    #Get HWND for the window to exclude
    try:
        hwnd = int(glfw.get_win32_window(window))
    except Exception:
        hwnd = 0

    #Apply clickthrough overlay styles
    _set_clickthrough_overlay_styles(hwnd)

    #Exclude overlay from capture to prevent feedback loop
    excluded = _attempt_exclude_from_capture(hwnd)
    print(f"[overlay] SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) success: {excluded}")
    if not excluded:
        print("[overlay] WARNING: Failed to exclude from capture. This may cause feedback loops (blurred blur).")

    #Register a global hotkey (Ctrl+Shift+Q) for graceful shutdown.
    _HOTKEY_ID = 1
    if not user32.RegisterHotKey(0, _HOTKEY_ID, MOD_CTRL | MOD_SHIFT, ord('Q')):
        print("[overlay] WARNING: Failed to register Ctrl+Shift+Q hotkey (already in use?). Use CTRL+C to stop.")
    else:
        print("[overlay] Press Ctrl+Shift+Q to stop the render loop gracefully.")

    #Prevent Windows from ghost-windowing this process.
    user32.DisableProcessWindowsGhosting()

    #Capture the first frame to determine settings like resolution and format for the blur renderer.
    print("[capture] Capturing first frame to determine resolution...")
    first = None
    while first is None and not glfw.window_should_close(window): #When we get frame number 1, we know the capture is working and can continue
        first = capture_desktop_excluding_hwnd(rgb=force_rgb)
        glfw.poll_events()
        time.sleep(0.02) #Brief sleep to avoid looping too much while waiting

    if first is None: #Failsafe: if we exit the loop without getting a frame, quit gracefully.
        print("[capture] No frame received; quitting.")
        glfw.destroy_window(window)
        glfw.terminate()
        return

    #Validate frame format
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        print(f"[capture] Unexpected frame format: {first.shape}")
        glfw.destroy_window(window)
        glfw.terminate()
        return

    #Select GL format based on channel count
    if capture_format == "bgr":
        input_format = GL_BGRA if first.shape[2] == 4 else GL_BGR
    else:
        input_format = GL_RGBA if first.shape[2] == 4 else GL_RGB

    cap_h, cap_w = first.shape[:2]
    print(f"[capture] Resolution: {cap_w}x{cap_h}")
    
    #Validate capture resolution matches physical monitor
    if phys_w is not None and phys_h is not None:
        if cap_w != phys_w or cap_h != phys_h:
            scale_x = phys_w / cap_w if cap_w > 0 else 1.0
            scale_y = phys_h / cap_h if cap_h > 0 else 1.0
            print(f"[WARNING] Capture mismatch: {cap_w}x{cap_h} != monitor {phys_w}x{phys_h} (scale: {scale_x:.2f}x)")

    #Set window size and pos
    glfw.set_window_size(window, overlay_size[0], overlay_size[1])
    glfw.set_window_pos(window, overlay_pos[0], overlay_pos[1])

    #Initialize blur renderer
    blur_renderer = GaussianBlurRenderer(cap_w, cap_h, str(blur_glsl_path), input_format=input_format)
    blur_renderer.set_params(radius_rgb=radius_rgb, sigma_rgb=sigma_rgb)
    blur_renderer.set_blur_active(blur_active)

    #Get the shader ready and set up the VAO for the fullscreen triangle. Simplest passthru shader possible
    display_prog = _create_display_shader()
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindVertexArray(0)
    tex_loc = glGetUniformLocation(display_prog, "uTexture")

    #No need for z buffering, the display is 2d and doesnt need depth test
    glDisable(GL_DEPTH_TEST)

    #FPS counter
    frames = 0
    last_fps_t = time.perf_counter()
    total_frames = 0  # absolute frame counter used for glFinish interval
    last_frame_delay_ms = 0.0  # capture-to-display latency of the most recent frame

    print(f"[session] participant_id={participant_id}, session_id={session_id}, blur_active={blur_active}")

    #Set Windows timer resolution to 1ms so time.sleep() is accurate for frame pacing.
    #Default Windows timer resolution is 15.625ms, which rounds every sleep up and causes ~35fps.
    ctypes.windll.winmm.timeBeginPeriod(1)

    #Render loop
    _msg = _MSG()  #Reused every frame, allocate once outside the loop
    try:
        while not glfw.window_should_close(window):
            t0 = time.perf_counter()
            t_capture_start = t0

            # 1)Capture a frame with dxgi_capture.py implementation
            frame = capture_desktop_excluding_hwnd(rgb=force_rgb)
            if frame is None: #Wait for frame if not available yet
                glfw.poll_events()
                time.sleep(0.001)
                continue

            #Validate frame format
            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"[capture] Unexpected frame format: {frame.shape}")
                glfw.poll_events()
                time.sleep(0.001)
                continue

            if frame.shape[2] == 4 and input_format not in (GL_BGRA, GL_RGBA):
                print("[capture] WARNING: Got 4-channel frame, but input_format is not GL_BGRA/GL_RGBA")

            if not frame.flags["C_CONTIGUOUS"]: #need contiguous array for glTexSubImage2D upload
                frame = np.ascontiguousarray(frame) 

            # 2)Process on GPU (separable blur passes).
            #Applies the blur shader to captured frame, returns outputted tex handle
            out_tex = blur_renderer.process(frame)
            _check_gl_error("after blur process") #posts error if the blur shader fails in OpenGL

            # 3)Display on overlay
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, fb_w, fb_h)

            #Clear to transparent black 
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)

            #Draw to the screen
            glUseProgram(display_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, out_tex)
            glUniform1i(tex_loc, 0)
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)
            _check_gl_error("after draw") #posts error if the display shader fails in OpenGL

            glfw.swap_buffers(window) #GPU syncs here, so rendered frame is presented after the call completes
            glFinish()  # ensure GPU has finished before measuring
            last_frame_delay_ms = (time.perf_counter() - t_capture_start) * 1000

            #Check for Ctrl+Shift+Q BEFORE polling and clearing w/ poll_events():
            if user32.PeekMessageW(ctypes.byref(_msg), None, WM_HOTKEY, WM_HOTKEY, 1):
                if _msg.wParam == _HOTKEY_ID:
                    print("[overlay] Ctrl+Shift+Q detected — shutting down.")
                    glfw.set_window_should_close(window, True)

            glfw.poll_events() #Poll for new events like window close or keypresses

            #FPS telemetry
            frames += 1
            total_frames += 1
            now = time.perf_counter()
            if now - last_fps_t >= 1.0:
                print(f"FPS: {frames}  (excluded_from_capture={excluded})")
                frames = 0
                last_fps_t = now

            #Print last known capture-to-display latency every 60 frames
            if total_frames % 60 == 0 and total_frames > 0:
                print(f"[latency] last frame delay: {last_frame_delay_ms:.1f}ms")

            #Periodic GPU latency measurement (when debug_gl_finish = True in settings)
            if debug_gl_finish and (total_frames % gl_finish_interval == 0):
                _t_gf = time.perf_counter()
                glFinish()
                print(f"[debug] glFinish frame {total_frames}: {(time.perf_counter()-_t_gf)*1000:.2f}ms")

            #Frame pacing, looking for target_fps from settings, and sleeps if rendering goes too fast
            #Avoids using 100% GPU on more FPS when it is not necessary
            dt = time.perf_counter() - t0
            target_dt = 1.0 / max(1, int(target_fps))
            if dt < target_dt:
                time.sleep(target_dt - dt)

    #Cleanup resources on exit
    finally:
        user32.UnregisterHotKey(0, _HOTKEY_ID)
        ctypes.windll.winmm.timeEndPeriod(1)
        blur_renderer.cleanup()
        glDeleteProgram(display_prog)
        glDeleteVertexArrays(1, [vao])
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
