"""
dxgi_capture.py (DXcam backend)

Uses DXcam (Desktop Duplication API)implementation written purely in python to capture frames.
DXcam respects WDA_EXCLUDEFROMCAPTURE in the desktop duplication path.
DXcam uses the desktop duplication API from Windows, and will thus respect the windows based flag.
Furthermore, DXcam has high performance, reaching 1080p240 on an Nvidia RTX 3090 GPU, with tabs and code running (according to author of DXcam)

Requirements:
  - Windows 10+
  - pip install dxcam numpy opencv-python

Notes:
  - DXcam returns frames as numpy arrays;
  - Arrays are output in RGB format when downstream needs RGB inputs
"""

#Imports needed
import time
from typing import Optional
import numpy as np
import cv2
import dxcam


class DXCamCapture:
    """DXcam-based capture that returns frames for CV processing."""
    def __init__(self, output_idx: int = 0, target_fps: Optional[int] = None):
        self.output_idx = output_idx
        self.target_fps = target_fps
        self.camera = None
        self._running = False

    def start(self) -> None:
        """Initialize DXcam and optionally start its capture thread."""
        self.camera = dxcam.create(output_idx=self.output_idx)
        if self.camera is None:
            raise RuntimeError("DXcam.create failed: no camera returned")
        if self.target_fps:
            #Video_mode ensures a smooth 60fps capture, even when idle or no new movement. Can cause performance overhead
            self.camera.start(target_fps=self.target_fps, video_mode=True) 
            self._running = True

    def get_frame(self, rgb: bool = False) -> Optional[np.ndarray]:
        """Capture a frame.

        Args:
            rgb: When True, convert BGR to RGB for downstream code expecting RGB.

        Returns:
            Numpy array for the frame, or None if no frame is available yet.
        """
        if self.camera is None:
            return None
        if self._running:
            frame = self.camera.get_latest_frame()
        else:
            frame = self.camera.grab()
        if frame is None:
            return None

        #DXcam returns BGR(A). Convert to RGB if flagged.
        if rgb:
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #A frame was obtained, return the np.ndarray
        return frame

    def stop(self) -> None:
        """Stop capture and release camera."""
        if self.camera is None:
            return
        if self._running:
            try:
                self.camera.stop()
            except Exception:
                pass
        self.camera = None
        self._running = False

_capture = None

def capture_desktop_excluding_hwnd(
    output_idx: int = 0,
    target_fps: Optional[int] = None,
    rgb: bool = False,
) -> Optional[np.ndarray]:
    """Capture a frame using DXcam.

    Uses Desktop Duplication under the hood and respects WDA_EXCLUDEFROMCAPTURE.

    Args:
        output_idx: Monitor index to capture.
        target_fps: If set, uses a background capture thread.
        rgb: Convert output to RGB.
    """
    global _capture

    if _capture is None:
        _capture = DXCamCapture(output_idx=output_idx, target_fps=target_fps)
        _capture.start()

    return _capture.get_frame(rgb=rgb)


def cleanup() -> None:
    """Clean up capture instance."""
    global _capture
    if _capture:
        _capture.stop()
        _capture = None


def test_capture_validity(output_idx: int = 0, num_frames: int = 10) -> bool:
    """
    Test whether DXcam is capturing meaningful data from the specified monitor.
    
    Checks:
    1. Monitor is accessible and DXcam initializes
    2. Frames are being captured (not None)
    3. Frames contain variance (not all one color/black)
    4. Resolution is sensible
    
    Returns:
        True if capture appears valid, False otherwise.
    """
    print(f"\n[test] Testing DXcam capture on output_idx={output_idx}...")
    
    try:
        camera = dxcam.create(output_idx=output_idx)
        if camera is None:
            print(f"[test] FAIL: dxcam.create returned None for output_idx={output_idx}")
            return False
        
        print(f"[test] DXcam initialized for monitor {output_idx}")
        
        # Get a few frames
        captured_frames = []
        for i in range(num_frames):
            frame = camera.grab()
            if frame is None:
                print(f"[test] FAIL: Frame {i+1} is None")
                camera.stop()
                return False
            captured_frames.append(frame)
            time.sleep(0.05)
        
        # Analyze the frames
        first_frame = captured_frames[0]
        print(f"[test] Resolution: {first_frame.shape[1]}x{first_frame.shape[0]}, dtype: {first_frame.dtype}, channels: {first_frame.shape[2]}")
        
        # Check if frame is all black or all one color
        unique_pixels = len(np.unique(first_frame.reshape(-1, first_frame.shape[2]), axis=0))
        print(f"[test] Unique colors in first frame: {unique_pixels}")
        
        if unique_pixels < 10:
            print(f"[test] WARNING: Very few unique colors ({unique_pixels}), possibly all black or corrupted")
        
        # Check variance across channels
        r_var = np.var(first_frame[:, :, 0])
        g_var = np.var(first_frame[:, :, 1])
        b_var = np.var(first_frame[:, :, 2])
        print(f"[test] Variance per channel - R: {r_var:.1f}, G: {g_var:.1f}, B: {b_var:.1f}")
        
        if r_var < 1 and g_var < 1 and b_var < 1:
            print(f"[test] FAIL: All channels have near-zero variance. Capture is likely all black or static.")
            camera.stop()
            return False
        
        # Check if any frame differs from the first
        frames_match = True
        for i, frame in enumerate(captured_frames[1:], start=2):
            if not np.array_equal(frame, first_frame):
                frames_match = False
                diff = np.abs(frame.astype(float) - first_frame.astype(float)).mean()
                print(f"[test] Frame {i} differs from frame 1 (avg diff: {diff:.1f})")
                break
        
        if frames_match:
            print(f"[test] WARNING: All frames are identical. Capture may be frozen.")
        
        camera.stop()
        print(f"[test] PASS: DXcam capture appears valid")
        return True
        
    except Exception as e:
        print(f"[test] FAIL: Exception during test: {e}")
        return False


if __name__ == "__main__":
    print("Testing DXcam Desktop Duplication...")
    
    # First, diagnose the capture
    valid = test_capture_validity(output_idx=0, num_frames=10)
    
    if not valid:
        print("\n[test] Capture validation failed. Check:")
        print("  1. Is the monitor connected and not in sleep mode?")
        print("  2. Try a different output_idx (0, 1, 2, etc.)")
        import sys
        sys.exit(1)
    
    print("\n[test] Running frame capture loop...")
    for i in range(5):
        start = time.perf_counter()
        frame = capture_desktop_excluding_hwnd(rgb=False)
        elapsed = time.perf_counter() - start

        if frame is not None:
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"Frame {i+1}: {frame.shape}, {elapsed*1000:.1f}ms, {fps:.1f} fps")
        else:
            print(f"Frame {i+1}: No frame yet")

        time.sleep(0.05)

    cleanup()
    print("Done!")
