"""
init_eyetracking.py

Unified eye tracking interface supporting Tobii Pro SDK hardware.

The sole purpose is to deliver a gazePos in NDC coords [0,1]² for the render loop.
This allows the eyetracking render overlay to determine foveal and peripheral regions,
applying Gaussian blurring only in the periphery based on gaze position.

Tobii Research API Used Directly:
==================================
- tr.find_all_eyetrackers()        → List[EyeTracker] (NATIVE)
- tracker.subscribe_to(...)        → Subscribe to data (NATIVE)
- tr.EYETRACKER_GAZE_DATA          → Subscription constant (NATIVE)
- GazeData.left_eye / right_eye    → EyeData (NATIVE)
- EyeData.gaze_point.position_on_display_area  → (x, y) [0,1] (NATIVE)
- EyeData.gaze_point.validity      → 0 for valid (NATIVE)

No custom EyeTracker/MouseTracker in tobii_research, so we wrap only for:
  1. Lock-free buffering of gaze data (callback pre-computes, render reads)
  2. Mouse fallback when hardware unavailable
"""

import glfw
import tobii_research as tr
import math
import threading
import time
from collections import deque
from typing import Any, Tuple


class TobiiGazeTracker:
    """Thin wrapper around native tr.EyeTracker with lock-free gaze buffering.
    
    The Tobii callback (_on_gaze_data) pre-computes normalized gaze position on the tracker
    thread. The render thread simply reads the cached result, avoiding lock contention."""
    
    def __init__(self, max_buffer_size: int = 120000):
        """Initialize (does not connect yet).

        Args:
            max_buffer_size: Maximum number of gaze samples to retain in memory.
                At 250 Hz, 120000 samples is roughly 8 minutes of data.
        """
        self.tracker = None  # Will be assigned tr.EyeTracker instance
        self.gaze_data = None  # Buffered native tobii_research.GazeData
        self.last_valid_position = (0.5, 0.5)  #Fallback during blinks
        self._cached_position = (0.5, 0.5)  # Cached position to avoid lock contention
        self._sample_lock = threading.Lock()
        self._gaze_samples = deque(maxlen=max(1000, int(max_buffer_size)))
        self._callback_durations_ms = deque(maxlen=max(1000, int(max_buffer_size)))
        self._last_callback_duration_ms = 0.0
    
    def initialize(self) -> bool:
        """Connect to Tobii hardware using tr.find_all_eyetrackers().
        
        Subscribes to native EYETRACKER_GAZE_DATA stream.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Use native tobii_research function
            eyetrackers = tr.find_all_eyetrackers()
            
            if not eyetrackers:
                print("[eyetracking] [ERROR] No Tobii eye trackers found")
                print("[eyetracking]   Ensure:")
                print("[eyetracking]   - Device is connected and powered")
                print("[eyetracking]   - Tobii Eye Tracker Manager is running")
                print("[eyetracking]   - Device drivers are installed")
                return False
            
            # Assign first native tr.EyeTracker instance
            self.tracker = eyetrackers[0]
            
            #Log device info from native tracker properties
            print(f"[eyetracking] [OK] Connected to: {self.tracker.model}")
            print(f"[eyetracking]   Serial: {self.tracker.serial_number}")
            print(f"[eyetracking]   Device: {self.tracker.device_name}")
            
            #Subscribe to gaze data stream
            self.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze_data)
            print("[eyetracking] [OK] Subscribed to native gaze data stream")
            
            return True
        
        except Exception as e:
            print(f"[eyetracking] [ERROR] Error initializing: {e}")
            return False
    
    @staticmethod
    def _is_valid_flag(value: Any) -> bool:
        """Normalize validity values from Tobii objects/dicts.

        Tobii's Python SDK exposes validity as booleans on high-level objects.
        Some integrations may still carry integer-like flags, so we coerce to bool.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return bool(value)

    @staticmethod
    def _safe_display_point(gaze_point: Any) -> tuple[float, float] | None:
        """Return a clamped normalized display point from Tobii gaze_point."""
        if gaze_point is None:
            return None
        raw_point = getattr(gaze_point, "position_on_display_area", None)
        if raw_point is None:
            return None

        # Native object form exposes tuples, but older wrappers may expose x/y.
        if isinstance(raw_point, tuple) and len(raw_point) >= 2:
            x_raw, y_raw = raw_point[0], raw_point[1]
        else:
            x_raw = getattr(raw_point, "x", None)
            y_raw = getattr(raw_point, "y", None)
        if x_raw is None or y_raw is None:
            return None

        try:
            x = float(x_raw)
            y = float(y_raw)
        except (TypeError, ValueError):
            return None
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _on_gaze_data(self, gaze_data):
        """Callback for native tr.EYETRACKER_GAZE_DATA stream.
        
        Called by tracker thread. Directly updates cached position without locks.
        In CPython, tuple assignment is atomic, so no synchronization needed.

        GazeData structure (from tobii_research):
          - left_eye: EyeData with gaze_point (x, y, validity)
          - right_eye: EyeData with gaze_point (x, y, validity)
          - system_time_stamp: int
        """
        callback_start = time.perf_counter()
        try:
            self.gaze_data = gaze_data

            left_eye = getattr(gaze_data, "left_eye", None)
            right_eye = getattr(gaze_data, "right_eye", None)
            left_gp = getattr(left_eye, "gaze_point", None) if left_eye is not None else None
            right_gp = getattr(right_eye, "gaze_point", None) if right_eye is not None else None

            left_valid = self._is_valid_flag(getattr(left_gp, "validity", False)) if left_gp is not None else False
            right_valid = self._is_valid_flag(getattr(right_gp, "validity", False)) if right_gp is not None else False

            left_xy = self._safe_display_point(left_gp) if left_valid else None
            right_xy = self._safe_display_point(right_gp) if right_valid else None

            valid_points = [p for p in [left_xy, right_xy] if p is not None]
            if valid_points:
                avg_x = sum(p[0] for p in valid_points) / len(valid_points)
                avg_y = sum(p[1] for p in valid_points) / len(valid_points)
                self._cached_position = (max(0.0, min(1.0, avg_x)), max(0.0, min(1.0, avg_y)))
                self.last_valid_position = self._cached_position

            left_pupil_data = getattr(left_eye, "pupil", None) if left_eye is not None else None
            right_pupil_data = getattr(right_eye, "pupil", None) if right_eye is not None else None
            left_pupil_valid = self._is_valid_flag(getattr(left_pupil_data, "validity", False)) if left_pupil_data is not None else False
            right_pupil_valid = self._is_valid_flag(getattr(right_pupil_data, "validity", False)) if right_pupil_data is not None else False

            left_pupil_mm = None
            if left_pupil_valid:
                try:
                    left_pupil_mm = float(getattr(left_pupil_data, "diameter", 0.0))
                except (TypeError, ValueError):
                    left_pupil_mm = None

            right_pupil_mm = None
            if right_pupil_valid:
                try:
                    right_pupil_mm = float(getattr(right_pupil_data, "diameter", 0.0))
                except (TypeError, ValueError):
                    right_pupil_mm = None

            system_ts = int(getattr(gaze_data, "system_time_stamp", 0) or 0)
            device_ts = int(getattr(gaze_data, "device_time_stamp", 0) or 0)
            sample = {
                "system_time_stamp": system_ts,
                "device_time_stamp": device_ts,
                "x": self._cached_position[0] if valid_points else None,
                "y": self._cached_position[1] if valid_points else None,
                "valid": bool(valid_points),
                "left_valid": left_valid,
                "right_valid": right_valid,
                "left_pupil_mm": left_pupil_mm,
                "right_pupil_mm": right_pupil_mm,
            }
            with self._sample_lock:
                self._gaze_samples.append(sample)
        except Exception:
            # Never allow callback exceptions to tear down the Tobii subscription thread.
            return
        finally:
            callback_duration_ms = (time.perf_counter() - callback_start) * 1000.0
            self._last_callback_duration_ms = callback_duration_ms
            with self._sample_lock:
                self._callback_durations_ms.append(callback_duration_ms)
    
    def get_gaze_position(self) -> Tuple[float, float]:
        """Get cached normalized gaze position (no locking overhead).
        
        The callback (_on_gaze_data) pre-computes the position on the tracker thread,
        so this is just a simple read - no locks needed.
        
        During blinking, Tobii marks data as invalid. Returns the last valid position
        instead of falling back to screen center.
        
        Returns:
            (x, y) in [0,1]², or last valid position if currently blinking
        """
        return self._cached_position

    def get_last_callback_duration_ms(self) -> float:
        """Return the most recent Tobii callback duration in milliseconds."""
        return float(self._last_callback_duration_ms)

    def get_callback_duration_stats(self) -> dict[str, float | int]:
        """Return min/max/avg/count for recorded Tobii callback durations."""
        with self._sample_lock:
            data = list(self._callback_durations_ms)
        if not data:
            return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "count": len(data),
            "min": min(data),
            "max": max(data),
            "avg": sum(data) / len(data),
        }

    def clear_gaze_samples(self) -> None:
        """Clear the buffered gaze samples."""
        with self._sample_lock:
            self._gaze_samples.clear()

    def get_gaze_samples(
        self,
        *,
        start_system_time_stamp: int | None = None,
        end_system_time_stamp: int | None = None,
        valid_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get buffered gaze samples with optional timestamp filtering.

        Time stamps are Tobii system clock microseconds (same unit as
        GazeData.system_time_stamp).
        """
        with self._sample_lock:
            out = list(self._gaze_samples)

        if start_system_time_stamp is not None:
            out = [s for s in out if int(s.get("system_time_stamp", 0)) >= int(start_system_time_stamp)]
        if end_system_time_stamp is not None:
            out = [s for s in out if int(s.get("system_time_stamp", 0)) <= int(end_system_time_stamp)]
        if valid_only:
            out = [s for s in out if bool(s.get("valid"))]
        return out

    def get_latest_system_time_stamp(self) -> int | None:
        """Return the latest buffered Tobii system timestamp, if available."""
        with self._sample_lock:
            if not self._gaze_samples:
                return None
            latest = self._gaze_samples[-1]
        try:
            ts = int(latest.get("system_time_stamp", 0) or 0)
        except Exception:
            return None
        return ts if ts > 0 else None

    @staticmethod
    def _group_contiguous_steps(
        steps: list[dict[str, Any]],
        *,
        max_inter_sample_gap_s: float,
    ) -> list[list[dict[str, Any]]]:
        if not steps:
            return []

        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = [steps[0]]
        gap_us = int(max(0.0, max_inter_sample_gap_s) * 1_000_000)

        for step in steps[1:]:
            prev = current_group[-1]
            same_label = step["label"] == prev["label"]
            contiguous = (int(step["t0"]) - int(prev["t1"])) <= gap_us
            if same_label and contiguous:
                current_group.append(step)
                continue
            groups.append(current_group)
            current_group = [step]

        groups.append(current_group)
        return groups

    @staticmethod
    def _detect_blinks_from_samples(
        samples: list[dict[str, Any]],
        *,
        min_blink_ms: float,
    ) -> list[dict[str, Any]]:
        """Infer blink-like events from runs where both eyes are invalid."""
        if not samples:
            return []

        ordered = sorted(samples, key=lambda s: int(s.get("system_time_stamp", 0)))
        blinks: list[dict[str, Any]] = []
        run_start_ts: int | None = None

        for sample in ordered:
            ts = int(sample.get("system_time_stamp", 0))
            both_invalid = not bool(sample.get("left_valid")) and not bool(sample.get("right_valid"))
            if both_invalid:
                if run_start_ts is None:
                    run_start_ts = ts
            elif run_start_ts is not None:
                dur_ms = (ts - run_start_ts) / 1000.0
                if dur_ms >= min_blink_ms:
                    blinks.append(
                        {
                            "start_system_time_stamp": run_start_ts,
                            "end_system_time_stamp": ts,
                            "duration_ms": round(dur_ms, 3),
                        }
                    )
                run_start_ts = None

        if run_start_ts is not None:
            end_ts = int(ordered[-1].get("system_time_stamp", run_start_ts))
            dur_ms = (end_ts - run_start_ts) / 1000.0
            if dur_ms >= min_blink_ms:
                blinks.append(
                    {
                        "start_system_time_stamp": run_start_ts,
                        "end_system_time_stamp": end_ts,
                        "duration_ms": round(dur_ms, 3),
                    }
                )
        return blinks

    def extract_eye_movement_events(
        self,
        *,
        samples: list[dict[str, Any]] | None = None,
        velocity_threshold_ndc_per_s: float = 1.0,
        min_fixation_ms: float = 180.0, #Fixation 180-275ms span
        min_saccade_ms: float = 20.0, #Min duration of saccade, 20-200ms
        max_saccade_ms: float = 120.0, #Max duration of saccade, 20-200ms
        min_blink_ms: float = 100.0, #Takes 300ms on average, but can be shorter
        max_inter_sample_gap_s: float = 0.075,
    ) -> dict[str, list[dict[str, Any]]]:
        """Classify buffered gaze into fixation/saccade events using I-VT.

        What constitutes a fixation is defined from Kar and Cocrcoran 2017
        
        Notes:
            - Velocity uses normalized display-area units per second.
            - Output is suitable for within-subject summaries and downstream ANOVA/GLMM.
            - Blink events are inferred from runs where both eyes are invalid.
        """
        source_samples = samples if samples is not None else self.get_gaze_samples()
        if not source_samples:
            return {"fixations": [], "saccades": [], "blinks": []}

        ordered = sorted(source_samples, key=lambda s: int(s.get("system_time_stamp", 0)))
        valid = [
            s for s in ordered
            if bool(s.get("valid")) and s.get("x") is not None and s.get("y") is not None
        ]

        steps: list[dict[str, Any]] = []
        for idx in range(1, len(valid)):
            prev = valid[idx - 1]
            curr = valid[idx]
            t0 = int(prev.get("system_time_stamp", 0))
            t1 = int(curr.get("system_time_stamp", 0))
            dt_s = (t1 - t0) / 1_000_000.0
            if dt_s <= 0:
                continue

            x0 = float(prev["x"])
            y0 = float(prev["y"])
            x1 = float(curr["x"])
            y1 = float(curr["y"])
            velocity = math.dist((x0, y0), (x1, y1)) / dt_s
            label = "saccade" if velocity >= velocity_threshold_ndc_per_s else "fixation"
            steps.append(
                {
                    "label": label,
                    "t0": t0,
                    "t1": t1,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "velocity": velocity,
                }
            )

        grouped = self._group_contiguous_steps(steps, max_inter_sample_gap_s=max_inter_sample_gap_s)
        fixations: list[dict[str, Any]] = []
        saccades: list[dict[str, Any]] = []

        for group in grouped:
            label = str(group[0]["label"])
            start_ts = int(group[0]["t0"])
            end_ts = int(group[-1]["t1"])
            duration_ms = (end_ts - start_ts) / 1000.0

            xs = [float(group[0]["x0"]) ] + [float(step["x1"]) for step in group]
            ys = [float(group[0]["y0"]) ] + [float(step["y1"]) for step in group]

            if label == "fixation":
                if duration_ms < min_fixation_ms:
                    continue
                fixations.append(
                    {
                        "start_system_time_stamp": start_ts,
                        "end_system_time_stamp": end_ts,
                        "duration_ms": round(duration_ms, 3),
                        "n_samples": len(xs),
                        "centroid_x": round(sum(xs) / len(xs), 6),
                        "centroid_y": round(sum(ys) / len(ys), 6),
                    }
                )
            else:
                if duration_ms < min_saccade_ms or duration_ms > max_saccade_ms:
                    continue
                peak_v = max(float(step["velocity"]) for step in group)
                mean_v = sum(float(step["velocity"]) for step in group) / len(group)
                amplitude = math.dist((xs[0], ys[0]), (xs[-1], ys[-1]))
                saccades.append(
                    {
                        "start_system_time_stamp": start_ts,
                        "end_system_time_stamp": end_ts,
                        "duration_ms": round(duration_ms, 3),
                        "n_steps": len(group),
                        "amplitude_ndc": round(amplitude, 6),
                        "peak_velocity_ndc_per_s": round(peak_v, 6),
                        "mean_velocity_ndc_per_s": round(mean_v, 6),
                    }
                )

        blinks = self._detect_blinks_from_samples(ordered, min_blink_ms=min_blink_ms)
        return {"fixations": fixations, "saccades": saccades, "blinks": blinks}

    def summarize_eye_movement_events(
        self,
        *,
        samples: list[dict[str, Any]] | None = None,
        velocity_threshold_ndc_per_s: float = 1.0,
        min_fixation_ms: float = 60.0,
        min_saccade_ms: float = 10.0,
        max_saccade_ms: float = 120.0,
        min_blink_ms: float = 60.0,
        max_inter_sample_gap_s: float = 0.075,
    ) -> dict[str, Any]:
        """Return trial/session-level metrics used by your analysis pipeline."""
        events = self.extract_eye_movement_events(
            samples=samples,
            velocity_threshold_ndc_per_s=velocity_threshold_ndc_per_s,
            min_fixation_ms=min_fixation_ms,
            min_saccade_ms=min_saccade_ms,
            max_saccade_ms=max_saccade_ms,
            min_blink_ms=min_blink_ms,
            max_inter_sample_gap_s=max_inter_sample_gap_s,
        )

        fixations = events["fixations"]
        saccades = events["saccades"]
        blinks = events["blinks"]

        fixation_durations = [float(ev["duration_ms"]) for ev in fixations]
        saccade_durations = [float(ev["duration_ms"]) for ev in saccades]
        blink_durations = [float(ev["duration_ms"]) for ev in blinks]

        return {
            "n_samples": len(samples) if samples is not None else len(self.get_gaze_samples()),
            "blinks_count": len(blinks),
            "blinks_total_duration_ms": round(sum(blink_durations), 3),
            "blinks_mean_duration_ms": round(sum(blink_durations) / len(blink_durations), 3) if blink_durations else None,
            "saccades_count": len(saccades),
            "saccades_total_duration_ms": round(sum(saccade_durations), 3),
            "saccades_mean_duration_ms": round(sum(saccade_durations) / len(saccade_durations), 3) if saccade_durations else None,
            "fixations_count": len(fixations),
            "fixations_total_duration_ms": round(sum(fixation_durations), 3),
            "fixations_mean_duration_ms": round(sum(fixation_durations) / len(fixation_durations), 3) if fixation_durations else None,
            "events": events,
        }

    def summarize_eye_movement_windows(
        self,
        windows: list[dict[str, Any]],
        *,
        velocity_threshold_ndc_per_s: float = 1.0,
        min_fixation_ms: float = 60.0,
        min_saccade_ms: float = 10.0,
        max_saccade_ms: float = 120.0,
        min_blink_ms: float = 60.0,
        max_inter_sample_gap_s: float = 0.075,
    ) -> list[dict[str, Any]]:
        """Summarize eye movement metrics for explicit trial/condition windows.

        Each window dict expects:
            - label: str
            - start_system_time_stamp: int
            - end_system_time_stamp: int
        """
        output: list[dict[str, Any]] = []
        for window in windows:
            start_ts = int(window.get("start_system_time_stamp", 0))
            end_ts = int(window.get("end_system_time_stamp", 0))
            label = str(window.get("label", "window"))
            cond = str(window.get("condition", ""))
            window_samples = self.get_gaze_samples(
                start_system_time_stamp=start_ts,
                end_system_time_stamp=end_ts,
                valid_only=False,
            )
            summary = self.summarize_eye_movement_events(
                samples=window_samples,
                velocity_threshold_ndc_per_s=velocity_threshold_ndc_per_s,
                min_fixation_ms=min_fixation_ms,
                min_saccade_ms=min_saccade_ms,
                max_saccade_ms=max_saccade_ms,
                min_blink_ms=min_blink_ms,
                max_inter_sample_gap_s=max_inter_sample_gap_s,
            )
            summary["label"] = label
            summary["condition"] = cond
            summary["start_system_time_stamp"] = start_ts
            summary["end_system_time_stamp"] = end_ts
            # Preserve optional trial/step metadata for downstream event-flagging.
            for key in ["section", "section_index", "phase", "step", "trial_index", "trial_file"]:
                if key in window:
                    summary[key] = window[key]
            output.append(summary)
        return output
    
    def calibrate(
        self,
        point_presenter=None,
        points=None,
        retries: int = 1,
        settle_time_s: float = 0.8,
    ) -> bool:
        """Run native Tobii screen-based calibration.

        Uses tobii_research.ScreenBasedCalibration and collects points in normalized
        display coordinates where (0,0) is top-left and (1,1) is bottom-right.

        Args:
            point_presenter: Optional callback called before each sample collection.
                Signature:
                  point_presenter(x, y, index, total, attempt) -> bool | None
                Return False to abort calibration.
            points: Optional sequence of normalized points [(x, y), ...].
                Defaults to a 5-point pattern suitable for Tobii Pro Fusion.
            retries: Number of additional passes for points that fail collection.
            settle_time_s: Time to wait at each point before collect_data().

        Returns:
            True if compute_and_apply succeeds, otherwise False.
        """
        if self.tracker is None:
            print("[eyetracking] [ERROR] Cannot calibrate: tracker not initialized")
            return False

        if point_presenter is None:
            print("[eyetracking] Calibration skipped: no point presenter was provided")
            print("[eyetracking] Use ParticipantTest calibration segment for interactive setup")
            return True

        if points is None:
            #Standard 5-point calibration pattern in display-area coordinates.
            points = [
                (0.50, 0.50),
                (0.10, 0.10),
                (0.90, 0.10),
                (0.10, 0.90),
                (0.90, 0.90),
            ]

        calibration = tr.ScreenBasedCalibration(self.tracker)
        points_to_collect = list(points)

        print("[eyetracking] Starting Tobii screen-based calibration...")

        try:
            calibration.enter_calibration_mode()

            for attempt in range(retries + 1):
                failed_points = []
                total = len(points_to_collect)

                for idx, (x, y) in enumerate(points_to_collect, start=1):
                    if point_presenter is not None:
                        should_continue = point_presenter(x, y, idx, total, attempt)
                        if should_continue is False:
                            print("[eyetracking] Calibration aborted by presenter callback")
                            return False

                    time.sleep(max(0.0, settle_time_s))
                    status = calibration.collect_data(float(x), float(y))

                    if status != tr.CALIBRATION_STATUS_SUCCESS:
                        failed_points.append((x, y))
                        print(f"[eyetracking]   collect_data failed at ({x:.2f}, {y:.2f})")

                if not failed_points:
                    break

                #Discarding failed points before retry is recommended by Tobii flow.
                for x, y in failed_points:
                    calibration.discard_data(float(x), float(y))

                points_to_collect = failed_points
                print(
                    f"[eyetracking] Retrying {len(points_to_collect)} failed points "
                    f"(attempt {attempt + 1}/{retries})"
                )

            result = calibration.compute_and_apply()
            if result.status == tr.CALIBRATION_STATUS_SUCCESS:
                print("[eyetracking] [OK] Calibration applied successfully")
                return True

            print(f"[eyetracking] [ERROR] Calibration apply failed: {result.status}")
            return False

        except Exception as e:
            print(f"[eyetracking] [ERROR] Calibration error: {e}")
            return False
        finally:
            try:
                calibration.leave_calibration_mode()
            except Exception as e:
                print(f"[eyetracking] [ERROR] Failed to leave calibration mode cleanly: {e}")
    
    def cleanup(self):
        """Unsubscribe from native tracker stream and cleanup."""
        if self.tracker is not None:
            try:
                self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA)
                print("[eyetracking] [OK] Disconnected")
            except Exception as e:
                print(f"[eyetracking] [ERROR] Cleanup error: {e}")
            finally:
                self.tracker = None


class MouseTracker:
    """Fallback: Mouse-based gaze (no native tobii_research equivalent)."""
    
    def __init__(self, window):
        self.window = window
    
    def initialize(self) -> bool:
        print("[eyetracking] [INFO] Using mouse (development mode, no Tobii hardware)")
        return True
    
    def get_gaze_position(self) -> Tuple[float, float]:
        """Get gaze from mouse cursor."""
        try:
            mx, my = glfw.get_cursor_pos(self.window)
            w, h = glfw.get_framebuffer_size(self.window)
            return (
                max(0.0, min(1.0, mx / max(1, w))),
                max(0.0, min(1.0, my / max(1, h)))
            )
        except:
            return (0.5, 0.5)

    def get_last_callback_duration_ms(self) -> float:
        return 0.0

    def get_callback_duration_stats(self) -> dict[str, float | int]:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    
    def calibrate(self) -> bool:
        return True
    
    def cleanup(self):
        pass


def initialize_eyetracker(window, gaze_source: str = "tobii"):
    """Initialize eye tracking: try native tobii_research hardware, fallback to mouse.
    
    Uses native tobii_research.find_all_eyetrackers() to detect hardware.
    
    Args:
        window: GLFW window
        gaze_source: "tobii" to attempt hardware (falls back to mouse if unavailable),
                     "mouse" to force mouse tracker (development/baseline mode)
    
    Returns:
        TobiiGazeTracker or MouseTracker instance
    """
    if gaze_source == "tobii":
        tracker = TobiiGazeTracker()
        if tracker.initialize():
            return tracker
        print("[eyetracking] Tobii not found, falling back to mouse")
    else:
        print(f"[eyetracking] gaze_source='{gaze_source}': using mouse tracker")
    mt = MouseTracker(window)
    mt.initialize()
    return mt


# Legacy interface
def get_gaze_pos(window, win_width: int, win_height: int, tracker=None) -> Tuple[float, float]:
    """Backward compatibility."""
    if tracker is None:
        tracker = MouseTracker(window)
    return tracker.get_gaze_position()
