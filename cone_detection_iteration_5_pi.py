"""Raspberry Pi 5 and Arducam cone slalom feedback — Iteration 5.

For a ribbon-connected CSI Arducam, frames come from Raspberry Pi OS through
Picamera2/libcamera. For a USB Arducam or laptop webcam, OpenCV capture remains
available as a fallback. Picamera2's RGB888 format is intentionally requested
because it produces the BGR byte order expected by OpenCV.

The detector, distance calibration, and camera-motion-aware slalom state
machine are reused from the earlier iterations. This program still provides
feedback only and does not command motors.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from cone_detection_iteration_1 import Detection, detect_cones, make_red_mask
from cone_detection_iteration_3 import (
    DEFAULT_CALIBRATION_FILE,
    CameraCalibration,
    make_dashboard,
)
from cone_detection_iteration_4 import CameraMotionSlalomNavigator


class CameraSource(Protocol):
    name: str

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def close(self) -> None: ...


class Picamera2Source:
    """Capture BGR-compatible frames from a Pi CSI camera."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        warmup_seconds: float,
    ) -> None:
        try:
            from libcamera import Transform
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is unavailable. Install it with: "
                "sudo apt install -y python3-picamera2"
            ) from error

        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError(
                "Picamera2 found no cameras. Check: rpicam-hello --list-cameras"
            )
        if not 0 <= camera_index < len(cameras):
            raise RuntimeError(
                f"Camera index {camera_index} is unavailable; detected {len(cameras)} camera(s)"
            )

        camera_info = cameras[camera_index]
        self.name = f"Picamera2 camera {camera_index}: {camera_info.get('Model', 'unknown')}"
        self._camera = Picamera2(camera_index)
        configuration = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=Transform(hflip=hflip, vflip=vflip),
            buffer_count=4,
        )
        self._camera.configure(configuration)
        self._camera.start()
        time.sleep(warmup_seconds)

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, self._camera.capture_array("main")
        except RuntimeError as error:
            print(f"Picamera2 capture failed: {error}")
            return False, None

    def close(self) -> None:
        try:
            self._camera.stop()
        finally:
            self._camera.close()


class OpenCVSource:
    """Capture from a USB Arducam or conventional webcam."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        warmup_seconds: float,
    ) -> None:
        self.name = f"OpenCV/USB camera {camera_index}"
        self._camera = cv2.VideoCapture(camera_index)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._hflip = hflip
        self._vflip = vflip
        if not self._camera.isOpened():
            raise RuntimeError(f"OpenCV could not open camera {camera_index}")
        time.sleep(warmup_seconds)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._camera.read()
        if not ok:
            return False, None
        if self._hflip or self._vflip:
            flip_code = -1 if self._hflip and self._vflip else (1 if self._hflip else 0)
            frame = cv2.flip(frame, flip_code)
        return True, frame

    def close(self) -> None:
        self._camera.release()


def create_camera(args: argparse.Namespace) -> CameraSource:
    common = {
        "camera_index": args.camera,
        "width": args.width,
        "height": args.height,
        "hflip": args.hflip,
        "vflip": args.vflip,
        "warmup_seconds": args.warmup_seconds,
    }
    if args.backend == "picamera2":
        try:
            return Picamera2Source(**common)
        except Exception as error:
            raise SystemExit(f"Could not start the CSI Arducam with Picamera2: {error}") from error
    if args.backend == "opencv":
        try:
            return OpenCVSource(**common)
        except Exception as error:
            raise SystemExit(f"Could not start the USB camera with OpenCV: {error}") from error

    try:
        return Picamera2Source(**common)
    except Exception as picamera_error:
        print(f"Picamera2 camera unavailable ({picamera_error}); trying USB/OpenCV.")
    try:
        return OpenCVSource(**common)
    except Exception as opencv_error:
        raise SystemExit(
            "No usable camera was found. Check `rpicam-hello --list-cameras` for a CSI "
            f"camera or /dev/video* for USB. OpenCV error: {opencv_error}"
        ) from opencv_error


def new_navigator(args: argparse.Namespace) -> CameraMotionSlalomNavigator:
    return CameraMotionSlalomNavigator(
        turn_start_cm=args.turn_start_cm,
        turn_start_height_ratio=args.turn_start_height_ratio,
        hard_turn_cm=args.hard_turn_cm,
        pass_distance_cm=args.pass_distance_cm,
        countersteer_frames=args.countersteer_frames,
        max_cones=getattr(args, "max_cones", None),
    )


def detections_for_dashboard(
    detections: list[Detection],
    current_target: Detection | None,
) -> list[Detection]:
    """Show every detected cone, with the navigation target drawn first."""
    if current_target is None:
        return detections
    return [current_target, *(item for item in detections if item is not current_target)]


def save_calibration(
    detection: Detection,
    frame_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> CameraCalibration:
    calibration = CameraCalibration.from_observation(
        detection,
        cone_height_cm=args.cone_height_cm,
        distance_cm=args.calibration_distance_cm,
        frame_shape=frame_shape,
    )
    calibration.save(args.calibration_file)
    print(
        f"Saved {args.calibration_file}: {args.cone_height_cm:.1f} cm cone at "
        f"{args.calibration_distance_cm:.1f} cm, focal length "
        f"{calibration.focal_length_px:.1f} px"
    )
    return calibration


def calibrate_from_multiple_frames(
    camera: CameraSource,
    args: argparse.Namespace,
) -> CameraCalibration:
    """Create a stable calibration without requiring a desktop window."""
    heights: list[int] = []
    frame_shape: tuple[int, ...] | None = None
    max_frames = 180
    required_detections = 30
    print(
        f"Calibrating: keep one full cone still at {args.calibration_distance_cm:.1f} cm..."
    )

    for _ in range(max_frames):
        ok, frame = camera.read()
        if not ok or frame is None:
            break
        frame_shape = frame.shape
        detections = detect_cones(frame, min_area=args.min_area)
        if len(detections) == 1 and not detections[0].cropped:
            heights.append(detections[0].height)
            if len(heights) >= required_detections:
                break

    if frame_shape is None or len(heights) < 10:
        raise SystemExit(
            "Calibration failed: fewer than 10 clean one-cone frames were found. "
            "Check the red mask and keep the full cone visible."
        )

    median_height = round(statistics.median(heights))
    representative = Detection(0, 0, 1, median_height, 1.0)
    print(f"Using median cone height {median_height} px from {len(heights)} frames.")
    return save_calibration(representative, frame_shape, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteration 5 Raspberry Pi 5 / Arducam cone slalom feedback"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="auto tries a CSI Picamera2 camera before USB/OpenCV",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450)
    parser.add_argument("--hflip", action="store_true", help="mirror image horizontally")
    parser.add_argument("--vflip", action="store_true", help="flip an upside-down camera")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", help="print feedback without a window")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="collect calibration frames, save calibration, and exit",
    )
    parser.add_argument("--cone-height-cm", type=float, default=30.5)
    parser.add_argument("--calibration-distance-cm", type=float, default=100.0)
    parser.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE
    )
    parser.add_argument("--turn-start-cm", type=float, default=130.0)
    parser.add_argument(
        "--turn-start-height-ratio",
        type=float,
        default=0.30,
        help="visual turn trigger as a fraction of image height",
    )
    parser.add_argument("--hard-turn-cm", type=float, default=80.0)
    parser.add_argument("--pass-distance-cm", type=float, default=60.0)
    parser.add_argument("--countersteer-frames", type=int, default=32)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(
        args.width,
        args.height,
        args.cone_height_cm,
        args.calibration_distance_cm,
        args.turn_start_cm,
        args.hard_turn_cm,
        args.pass_distance_cm,
        args.warmup_seconds,
    ) <= 0:
        raise SystemExit("Image sizes, times, and physical measurements must be positive")
    if not args.turn_start_cm > args.hard_turn_cm > args.pass_distance_cm:
        raise SystemExit("Distances must satisfy: turn-start > hard-turn > pass-distance")
    if not 0.0 < args.turn_start_height_ratio < 1.0:
        raise SystemExit("turn-start-height-ratio must be between 0 and 1")
    if args.countersteer_frames < 1:
        raise SystemExit("countersteer-frames must be at least 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    camera = create_camera(args)
    print(f"Using {camera.name} at requested size {args.width}x{args.height}")

    try:
        if args.calibrate:
            calibrate_from_multiple_frames(camera, args)
            return

        calibration = CameraCalibration.load(args.calibration_file)
        if args.headless and calibration is None:
            raise SystemExit(
                "Headless mode needs distance calibration. Put the cone at the measured "
                "mark and run again with --calibrate."
            )

        navigator = new_navigator(args)
        if calibration is None:
            print("No calibration found. In the dashboard, position the cone and press C.")
        else:
            print(f"Loaded distance calibration from {args.calibration_file}")
        print("Iteration 5 running. C=calibrate, R=reset, Q/Escape=stop, Ctrl+C=headless stop.")

        previous_feedback: str | None = None
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("The camera stopped returning frames.")
                break

            all_detections = detect_cones(frame, min_area=args.min_area)
            feedback = navigator.update(all_detections, calibration, frame.shape)
            if args.headless:
                if feedback != previous_feedback:
                    print(feedback)
                    previous_feedback = feedback
                continue

            mask = make_red_mask(frame)
            display_detections = detections_for_dashboard(
                all_detections,
                navigator.current_target,
            )
            dashboard = make_dashboard(
                frame,
                mask,
                display_detections,
                navigator,
                feedback,
                navigator.smoothed_distance_cm,
                calibration,
                args.calibration_distance_cm,
                display_width=args.display_width,
            )
            cv2.imshow("Pi 5 Arducam Cone Slalom - Iteration 5", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                navigator = new_navigator(args)
            if key == ord("c"):
                if not all_detections:
                    print("Calibration failed: no cone is detected.")
                elif all_detections[0].cropped:
                    print("Calibration failed: keep the full cone inside the image.")
                else:
                    calibration = save_calibration(all_detections[0], frame.shape, args)
                    navigator = new_navigator(args)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
