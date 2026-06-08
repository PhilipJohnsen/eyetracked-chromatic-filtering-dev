"""
eyetracking-render-loop.py

Goal:
  Continuous click-through overlay that renders a foveated blurred version of the desktop
  Blur is applied only in the periphery based on gaze position
  Currently uses mouse position as gaze proxy
  Utilises the dxgi_capture.py implementation for high performance capture (uses DXcam import)
  Applies a gaussian blur with given kernel for each colour channel

Implementation:
  1) Create an always-on-top GLFW window that is:
        borderless
        transparent framebuffer capable
        click-through via Win32 extended styles
  2) Exclude the overlay itself from capture
     SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE).
     Correctness claim:
       - If the capture path honors the affinity flag, the captured frames
         will not include our overlay => no feedback loop.
         Per DXcam implementation using desktop duplication API from Windows, this flag should be respected
  3) Track gaze position (currently mouse position) and convert to normalized coordinates
  4) Render loop:
       capture_desktop_excluding_hwnd() -> foveated_blur_renderer.process(frame, gaze_pos) -> draw overlay.
     Correctness claim:
       - We apply the filter exactly once per fresh captured frame.
       - Blur is only applied in peripheral regions based on distance from gaze.
"""

# -----------------------------
# DPI Awareness - Must execute BEFORE any imports that query screen dimensions
# -----------------------------
import ctypes

def _set_dpi_awareness():
    """Set DPI awareness using multiple fallback methods for compatibility."""
    # Try shcore.SetProcessDpiAwareness (Windows 8.1+)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return True
    except (AttributeError, OSError):
        pass
    
    # Try user32.SetProcessDpiAwarenessContext (Windows 10 1607+)
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return True
    except (AttributeError, OSError):
        pass
    
    # Try user32.SetProcessDPIAware (Windows Vista+)
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            return True
    except (AttributeError, OSError):
        pass
    
    return False

# Execute DPI awareness before any screen queries
if not _set_dpi_awareness():
    print("[WARNING] Failed to set DPI awareness - capture may use logical pixels instead of physical")

# ---------------------
# Imports
# ---------------------
import csv
import time
from pathlib import Path
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from dxgi_capture import capture_desktop_excluding_hwnd
from render_foveated_blur import FoveatedBlurRenderer
from utility.load_settings import load_settings, unpack_settings
from init_eyetracking import initialize_eyetracker


# -----------------------------
# Win32 helpers window styles. Specify transparency for clicks, keep above other windows etc.
# -----------------------------
user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
GWL_STYLE   = -16
# Window styles
WS_EX_LAYERED      = 0x00080000
WS_EX_TRANSPARENT  = 0x00000020  # "hit-test transparent" => click-through
WS_EX_TOOLWINDOW   = 0x00000080  # do not show in alt-tab
WS_EX_TOPMOST      = 0x00000008  # keep above normal windows
# Normal window styles
WS_POPUP = 0x80000000
# Layered attributes
LWA_ALPHA = 0x00000002
# Display affinity:
# WDA_EXCLUDEFROMCAPTURE = 0x11 (from Win10 2004+)
WDA_EXCLUDEFROMCAPTURE = 0x00000011

#Hotkey modifiers and message
MOD_CTRL  = 0x0002
MOD_SHIFT = 0x0004
WM_HOTKEY = 0x0312

#For checking the message queue for hotkey presses
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

    # Set popup style (borderless)
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~0x00CF0000
    style |= WS_POPUP
    user32.SetWindowLongW(hwnd, GWL_STYLE, style)
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ex |= (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)

    # Set layered window to fully opaque
    user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)

    # Force style update
    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0020 | 0x0040)  # NOSIZE|NOMOVE|FRAMECHANGED|NOACTIVATE


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
        width = ctypes.windll.user32.GetSystemMetrics(0)   # SM_CXSCREEN
        height = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
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


# Used for debugging capture issues, verify format/strides/contiguity of captured frames.
def _check_gl_error(tag: str) -> None:
    """Print any GL errors and clear the error flag."""
    err = glGetError()
    if err == GL_NO_ERROR:
        return
    while err != GL_NO_ERROR:
        print(f"[gl] error {tag}: 0x{err:04x}")
        err = glGetError()


def _get_normalized_gaze_position(window, win_width: int, win_height: int) -> tuple:
    """Get gaze position in normalized coordinates [0,1] x [0,1].
    
    Currently uses mouse position as proxy for gaze.
    (0, 0) = top-left corner
    (1, 1) = bottom-right corner
    
    Args:
        window: GLFW window handle
        win_width: Window width in pixels
        win_height: Window height in pixels
    
    Returns:
        (x, y) tuple in normalized coordinates [0,1]
    """
    #Get mouse pos
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    
    #Clamp to window
    mouse_x = max(0, min(mouse_x, win_width))
    mouse_y = max(0, min(mouse_y, win_height))
    #Normalize to [0, 1]
    norm_x = mouse_x / max(1, win_width)
    norm_y = mouse_y / max(1, win_height)
    
    return (norm_x, norm_y)

# -----------------------------
# Main render-loop
# -----------------------------
def main():
    #Settings from utility/settings.txt (relative to this script)
    settings_path = Path(__file__).parent / "utility" / "settings.txt"
    settings = load_settings(str(settings_path))
    (target_fps, force_rgb, capture_format, 
     debug_gl_finish, gl_finish_interval, 
     overlay_size, overlay_pos, radius_rgb, sigma_rgb, shader_path,
     foveal_radius, transition_width,
     gaze_source, blur_active, participant_id, session_id,
     log_gaze, log_path, lum_correction) = unpack_settings(settings)
    
    #Blur shader path (relative to this script, from settings)
    blur_glsl_path = Path(__file__).parent / shader_path
    
    #Foveated compositing shader path
    composite_glsl_path = Path(__file__).parent / "shader" / "foveal_composite.glsl"

    #Query physical monitor resolution for validation
    phys_w, phys_h = _get_physical_monitor_resolution()

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # GLFW specifics:
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)  # borderless
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # always on top
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)  # allow alpha in framebuffer for proper compositing
    glfw.window_hint(glfw.FOCUSED, glfw.FALSE)  # Don't steal focus from other windows
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)  # Allow resizing in case the user wants to tweak this

    # Create window context
    window = glfw.create_window(overlay_size[0], overlay_size[1], 
                                 "Chromatic Filtering Overlay (eyetracking)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # we select the capture pacing via settings, thus disable vsync 

    # Get HWND for the window exclude flag
    try:
        hwnd = int(glfw.get_win32_window(window))
    except Exception:
        hwnd = 0

    # Apply clickthrough overlay styles
    _set_clickthrough_overlay_styles(hwnd)

    # Exclude overlay from capture to prevent feedback loop
    excluded = _attempt_exclude_from_capture(hwnd)
    print(f"[overlay] SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) success: {excluded}")
    if not excluded:
        print("[overlay] WARNING: Failed to exclude from capture. This may cause feedback loops (blurred blur).")

    #Register a global hotkey (Ctrl+Shift+Q) for graceful shutdown.
    #RegisterHotKey posts WM_HOTKEY to this thread's queue
    #regardless of which window has focus.
    _HOTKEY_ID = 1
    if not user32.RegisterHotKey(0, _HOTKEY_ID, MOD_CTRL | MOD_SHIFT, ord('Q')):
        print("[overlay] WARNING: Failed to register Ctrl+Shift+Q hotkey (already in use?). Use CTRL+C to stop.")
    else:
        print("[overlay] Press Ctrl+Shift+Q to stop the render loop gracefully.")

    #Prevent Windows from ghost-windowing this process.
    user32.DisableProcessWindowsGhosting()

    # Initialize eye tracking (gaze_source from settings: "tobii" or "mouse")
    tracker = initialize_eyetracker(window, gaze_source=gaze_source)
    tracker.calibrate()

    # Capture the first frame to determine settings like resolution and format for the blur renderer.
    print("[capture] Capturing first frame to determine resolution...")
    first = None
    while first is None and not glfw.window_should_close(window):  # When we get frame number 1, we know the capture is working and can continue
        first = capture_desktop_excluding_hwnd(rgb=force_rgb)
        glfw.poll_events()
        time.sleep(0.02)  # Brief sleep to avoid looping too much while waiting

    if first is None:  # Failsafe: if we exit the loop without getting a frame, quit gracefully.
        print("[capture] No frame received; quitting.")
        glfw.destroy_window(window)
        glfw.terminate()
        return

    # Validate frame format
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        print(f"[capture] Unexpected frame format: {first.shape}")
        glfw.destroy_window(window)
        glfw.terminate()
        return

    # Select GL format based on channel count
    if capture_format == "bgr":
        input_format = GL_BGRA if first.shape[2] == 4 else GL_BGR
    else:
        input_format = GL_RGBA if first.shape[2] == 4 else GL_RGB

    cap_h, cap_w = first.shape[:2]
    print(f"[capture] Resolution: {cap_w}x{cap_h}")
    
    # Validate capture resolution matches physical monitor
    if phys_w is not None and phys_h is not None:
        if cap_w != phys_w or cap_h != phys_h:
            scale_x = phys_w / cap_w if cap_w > 0 else 1.0
            scale_y = phys_h / cap_h if cap_h > 0 else 1.0
            print(f"[WARNING] Capture mismatch: {cap_w}x{cap_h} != monitor {phys_w}x{phys_h} (scale: {scale_x:.2f}x)")

    # Set window size and pos
    glfw.set_window_size(window, overlay_size[0], overlay_size[1])
    glfw.set_window_pos(window, overlay_pos[0], overlay_pos[1])

    # Initialize foveated blur renderer
    foveated_renderer = FoveatedBlurRenderer(
        cap_w, cap_h, 
        str(blur_glsl_path), 
        str(composite_glsl_path),
        input_format=input_format
    )
    foveated_renderer.set_blur_params(radius_rgb=radius_rgb, sigma_rgb=sigma_rgb)
    aspect_ratio = overlay_size[0] / max(1, overlay_size[1])
    foveated_renderer.set_foveal_params(foveal_radius=foveal_radius, transition_width=transition_width,
                                        aspect_ratio=aspect_ratio)
    foveated_renderer.set_blur_active(blur_active)
    foveated_renderer.set_lum_correction(lum_correction)

    # Get the shader ready and set up the VAO for the fullscreen triangle. Simplest passthru shader possible
    display_prog = _create_display_shader()
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindVertexArray(0)
    tex_loc = glGetUniformLocation(display_prog, "uTexture")

    # No need for z buffering, the display is 2d and doesn't need depth test
    glDisable(GL_DEPTH_TEST)

    # FPS counter
    frames = 0
    last_fps_t = time.perf_counter()
    total_frames = 0  # absolute frame counter used for glFinish interval
    last_frame_delay_ms = 0.0  # capture-to-display latency of the most recent frame

    print(f"[eyetracking] Foveal radius: {foveal_radius:.2f}, Transition width: {transition_width:.2f}")
    print(f"[session] participant_id={participant_id}, session_id={session_id}, blur_active={blur_active}")

    # Set Windows timer resolution to 1ms so time.sleep() is accurate for frame pacing.
    # Default Windows timer resolution is 15.625ms, which rounds every sleep up and causes ~35fps.
    ctypes.windll.winmm.timeBeginPeriod(1)

    # Gaze logging setup (only when log_gaze = True in settings)
    _gaze_log_file = None
    _gaze_log_writer = None
    if log_gaze:
        _log_path = Path(__file__).parent / log_path
        _gaze_log_file = open(_log_path, 'w', newline='')
        _gaze_log_writer = csv.writer(_gaze_log_file)
        _gaze_log_writer.writerow(['timestamp_ms', 'frame', 'gaze_x', 'gaze_y'])
        print(f"[session] Logging gaze to {_log_path}")

    #Render loop
    _msg = _MSG()  #Reused every frame, allocate once
    try:
        while not glfw.window_should_close(window):
            t0 = time.perf_counter()
            t_capture_start = t0

            # 1) Capture a frame with dxgi_capture.py implementation
            frame = capture_desktop_excluding_hwnd(rgb=force_rgb)
            if frame is None:  # Wait for frame if not available yet
                glfw.poll_events()
                time.sleep(0.001)
                continue

            # Validate frame format
            if frame.ndim != 3 or frame.shape[2] !=3: #never expect 4 channels
                print(f"[capture] Unexpected frame format: {frame.shape}")
                glfw.poll_events()
                time.sleep(0.001)
                continue

            if frame.shape[2] == 4 and input_format not in (GL_BGRA, GL_RGBA):
                print("[capture] WARNING: Got 4-channel frame, but input_format is not GL_BGRA/GL_RGBA")

            if not frame.flags["C_CONTIGUOUS"]:  # need contiguous array for glTexSubImage2D upload
                frame = np.ascontiguousarray(frame) 

            # 2) Get gaze position from tracker (Tobii hardware or mouse fallback)
            gaze_pos = tracker.get_gaze_position()

            # Log gaze position with frame timestamp (when log_gaze = True in settings)
            if log_gaze and _gaze_log_writer:
                _gaze_log_writer.writerow([round(t0 * 1000), total_frames,
                                           f"{gaze_pos[0]:.6f}", f"{gaze_pos[1]:.6f}"])

            # 3) Process on GPU (foveated blur: sharp in fovea, blurred in periphery)
            # Applies the blur shader to captured frame, then composites based on eccentricity
            out_tex = foveated_renderer.process(frame, gaze_pos=gaze_pos)
            _check_gl_error("after foveated process")  # posts error if the blur shader fails in OpenGL

            # 4) Display on overlay
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, fb_w, fb_h)

            # Clear to transparent black 
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)

            # Draw to the screen
            glUseProgram(display_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, out_tex)
            glUniform1i(tex_loc, 0)
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)
            _check_gl_error("after draw")  # posts error if the display shader fails in OpenGL

            glfw.swap_buffers(window)  # GPU syncs here, so rendered frame is presented after the call completes
            glFinish()  # ensure GPU has finished before measuring
            last_frame_delay_ms = (time.perf_counter() - t_capture_start) * 1000

            # Check for Ctrl+Shift+Q BEFORE polling and clearing w/ poll_events():

            if user32.PeekMessageW(ctypes.byref(_msg), None, WM_HOTKEY, WM_HOTKEY, 1):
                if _msg.wParam == _HOTKEY_ID:
                    print("[overlay] Ctrl+Shift+Q detected — shutting down.")
                    glfw.set_window_should_close(window, True)

            glfw.poll_events()  # Poll for new events like window close or keypresses

            # FPS telemetry
            frames += 1
            total_frames += 1
            now = time.perf_counter()
            if now - last_fps_t >= 1.0:
                print(f"FPS: {frames}  (gaze: {gaze_pos[0]:.3f}, {gaze_pos[1]:.3f})")
                frames = 0
                last_fps_t = now

            # Print last known capture-to-display latency every 60 frames
            if total_frames % 60 == 0 and total_frames > 0:
                print(f"[latency] last frame delay: {last_frame_delay_ms:.1f}ms")

            # Periodic GPU latency measurement (when debug_gl_finish = True in settings)
            if debug_gl_finish and (total_frames % gl_finish_interval == 0):
                _t_gf = time.perf_counter()
                glFinish()
                print(f"[debug] glFinish frame {total_frames}: {(time.perf_counter()-_t_gf)*1000:.2f}ms")

            # Frame pacing, looking for target_fps from settings, and sleeps if rendering goes too fast
            # Avoids using 100% GPU on more FPS when it is not necessary
            dt = time.perf_counter() - t0
            target_dt = 1.0 / max(1, int(target_fps))
            if dt < target_dt:
                time.sleep(target_dt - dt)

    # Cleanup resources on exit
    finally:
        user32.UnregisterHotKey(0, _HOTKEY_ID)
        ctypes.windll.winmm.timeEndPeriod(1)
        if _gaze_log_file:
            _gaze_log_file.close()
        tracker.cleanup()
        foveated_renderer.cleanup()
        glDeleteProgram(display_prog)
        glDeleteVertexArrays(1, [vao])
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
