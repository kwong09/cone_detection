"""Distance-aware cone detection and head-on slalom feedback — Iteration 3.

The camera is calibrated from a cone of known height at a measured distance.
The resulting focal length lets the program estimate distance from the cone's
pixel height. Slalom feedback approaches each cone head-on, delays the planned
turn until the configured distance, and confirms that the cone passes on the
correct side before alternating direction.

This program provides steering feedback only. It does not command motors yet.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cone_detection_iteration_1 import (
    Detection,
    detect_cones,
    draw_detections,
    draw_panel_title,
    make_red_mask,
)


DEFAULT_CALIBRATION_FILE = Path(__file__).with_name("cone_camera_calibration.json")


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole-camera calibration for cones of one known physical height."""

    focal_length_px: float
    cone_height_cm: float
    frame_width: int
    frame_height: int
    calibration_distance_cm: float

    @classmethod
    def from_observation(
        cls,
        detection: Detection,
        cone_height_cm: float,
        distance_cm: float,
        frame_shape: tuple[int, ...],
    ) -> "CameraCalibration":
        if detection.height <= 0 or cone_height_cm <= 0 or distance_cm <= 0:
            raise ValueError("Cone height, pixel height, and distance must be positive")
        focal_length_px = detection.height * distance_cm / cone_height_cm
        return cls(
            focal_length_px=focal_length_px,
            cone_height_cm=cone_height_cm,
            frame_width=frame_shape[1],
            frame_height=frame_shape[0],
            calibration_distance_cm=distance_cm,
        )

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                focal_length_px=float(data["focal_length_px"]),
                cone_height_cm=float(data["cone_height_cm"]),
                frame_width=int(data["frame_width"]),
                frame_height=int(data["frame_height"]),
                calibration_distance_cm=float(data["calibration_distance_cm"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Ignoring invalid calibration file {path}: {error}")
            return None

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "focal_length_px": self.focal_length_px,
                    "cone_height_cm": self.cone_height_cm,
                    "frame_width": self.frame_width,
                    "frame_height": self.frame_height,
                    "calibration_distance_cm": self.calibration_distance_cm,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def estimate_distance_cm(
        self, detection: Detection, frame_shape: tuple[int, ...]
    ) -> float:
        if detection.height <= 0:
            return math.inf
        # Pixel focal length scales with image height when the camera keeps the
        # same field of view. Recalibrate if a new sensor mode crops the image.
        scaled_focal_px = self.focal_length_px * frame_shape[0] / self.frame_height
        return self.cone_height_cm * scaled_focal_px / detection.height


@dataclass
class DistanceAwareSlalomNavigator:
    """Use range and expected image motion to time alternating cone passes."""

    turn_start_cm: float = 130.0
    hard_turn_cm: float = 80.0
    pass_distance_cm: float = 60.0
    direction_index: int = 0
    cones_passed: int = 0
    phase: str = "SEARCHING"
    pass_armed: bool = False
    candidate_frames: int = 0
    largest_target_area: int = 0
    closest_distance_cm: float = math.inf
    smoothed_distance_cm: float | None = None
    pass_confirmation_frames: int = 5

    @property
    def direction(self) -> str:
        return ("RIGHT", "LEFT")[self.direction_index]

    @property
    def next_direction(self) -> str:
        return ("LEFT", "RIGHT")[self.direction_index]

    @property
    def expected_cone_side(self) -> str:
        return "LEFT" if self.direction == "RIGHT" else "RIGHT"

    def _cone_is_on_pass_side(self, center_x_ratio: float) -> bool:
        if self.direction == "RIGHT":
            return center_x_ratio <= 0.40
        return center_x_ratio >= 0.60

    def _finish_pass(
        self, new_target: Detection | None, new_distance_cm: float | None
    ) -> None:
        self.cones_passed += 1
        self.direction_index = 1 - self.direction_index
        self.pass_armed = False
        self.candidate_frames = 0
        self.largest_target_area = 0
        self.closest_distance_cm = math.inf
        self.smoothed_distance_cm = None
        self.phase = "APPROACHING" if new_target is not None else "SEARCHING"

        if new_target is not None and new_distance_cm is not None:
            self.largest_target_area = new_target.width * new_target.height
            self.closest_distance_cm = new_distance_cm
            self.smoothed_distance_cm = new_distance_cm

    def update(
        self,
        detections: list[Detection],
        raw_distance_cm: float | None,
        frame_shape: tuple[int, ...],
        calibrated: bool,
    ) -> str:
        """Advance the navigation state and return human-readable feedback."""
        if not calibrated:
            self.phase = "CALIBRATION REQUIRED"
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"

        target = detections[0] if detections else None
        if target is not None and raw_distance_cm is not None:
            area = target.width * target.height
            center_x_ratio = target.center[0] / frame_shape[1]
            self.largest_target_area = max(self.largest_target_area, area)
            self.closest_distance_cm = min(self.closest_distance_cm, raw_distance_cm)

            # Smooth normal bbox jitter, but keep the raw value for recognizing
            # the next, much farther cone after the current cone passes.
            if self.smoothed_distance_cm is None:
                self.smoothed_distance_cm = raw_distance_cm
            else:
                self.smoothed_distance_cm = (
                    0.70 * self.smoothed_distance_cm + 0.30 * raw_distance_cm
                )

            if self.phase in {"SEARCHING", "CALIBRATION REQUIRED"}:
                self.phase = "APPROACHING"

            if (
                self.phase == "APPROACHING"
                and self.smoothed_distance_cm <= self.turn_start_cm
            ):
                self.phase = "TURNING"

            on_pass_side = self._cone_is_on_pass_side(center_x_ratio)
            if (
                self.phase == "TURNING"
                and self.smoothed_distance_cm <= self.pass_distance_cm
                and on_pass_side
            ):
                self.phase = "PASSING"
                self.pass_armed = True

            new_far_target = (
                self.pass_armed
                and raw_distance_cm
                > max(self.turn_start_cm, self.closest_distance_cm * 1.70)
                and area < self.largest_target_area * 0.48
            )
            self.candidate_frames = self.candidate_frames + 1 if new_far_target else 0
        elif self.pass_armed:
            # The correctly positioned close cone has now left the frame.
            self.candidate_frames += 1
        else:
            # Disappearance before the cone reaches the passing side is unsafe
            # and must never be counted as a completed pass.
            self.candidate_frames = 0

        if self.pass_armed and self.candidate_frames >= self.pass_confirmation_frames:
            self._finish_pass(target, raw_distance_cm)

        return self.feedback(detections, calibrated, frame_shape)

    def feedback(
        self,
        detections: list[Detection],
        calibrated: bool,
        frame_shape: tuple[int, ...],
    ) -> str:
        if not calibrated:
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"
        if not detections:
            if self.pass_armed:
                return f"HOLD {self.direction} - FINISH PASS"
            if self.phase == "TURNING":
                return "STOP - CONE LOST BEFORE PASS"
            return "STOP - FIND NEXT CONE"

        target = detections[0]
        center_x_ratio = target.center[0] / frame_shape[1]
        distance = self.smoothed_distance_cm
        if distance is None:
            return "STOP - DISTANCE UNAVAILABLE"

        distance_text = format_distance(distance)
        if self.pass_armed:
            return f"HOLD {self.direction} - PASSING AT {distance_text}"

        if distance > self.turn_start_cm:
            # Center on the next cone while it is still far away so that the
            # planned alternating turn starts from a predictable head-on line.
            if center_x_ratio < 0.44:
                return f"ALIGN LEFT - CONE {distance_text}"
            if center_x_ratio > 0.56:
                return f"ALIGN RIGHT - CONE {distance_text}"
            return f"FORWARD - CONE {distance_text}"

        on_pass_side = self._cone_is_on_pass_side(center_x_ratio)
        if distance <= self.hard_turn_cm and not on_pass_side:
            return f"HARD {self.direction} - CONE {distance_text}"
        if on_pass_side:
            return f"HOLD {self.direction} - CONE MOVING {self.expected_cone_side}"
        return f"BEGIN {self.direction} TURN - CONE {distance_text}"


def format_distance(distance_cm: float | None) -> str:
    if distance_cm is None or not math.isfinite(distance_cm):
        return "--"
    return f"{distance_cm:.0f} cm / {distance_cm / 2.54:.0f} in"


def draw_distance(camera_panel: np.ndarray, detection: Detection, distance_cm: float | None) -> None:
    if distance_cm is None:
        return
    label = f"DISTANCE: {format_distance(distance_cm)}"
    label_y = detection.y + detection.height + 28
    if label_y >= camera_panel.shape[0] - 8:
        label_y = max(30, detection.y + 32)
    cv2.putText(
        camera_panel,
        label,
        (detection.x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        5,
    )
    cv2.putText(
        camera_panel,
        label,
        (detection.x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 0),
        2,
    )


def make_dashboard(
    frame: np.ndarray,
    mask: np.ndarray,
    detections: list[Detection],
    navigator: DistanceAwareSlalomNavigator,
    feedback: str,
    distance_cm: float | None,
    calibration: CameraCalibration | None,
    calibration_distance_cm: float,
    display_width: int = 1600,
) -> np.ndarray:
    camera_panel = draw_detections(frame, detections)
    if detections:
        draw_distance(camera_panel, detections[0], distance_cm)
    mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    panel_width = max(320, display_width // 2)
    panel_height = max(180, round(frame.shape[0] * panel_width / frame.shape[1]))
    panel_size = (panel_width, panel_height)
    camera_panel = cv2.resize(camera_panel, panel_size, interpolation=cv2.INTER_AREA)
    mask_panel = cv2.resize(mask_panel, panel_size, interpolation=cv2.INTER_NEAREST)
    draw_panel_title(camera_panel, "CAMERA + DISTANCE")
    draw_panel_title(mask_panel, "RED CONE MASK")

    panels = np.hstack((camera_panel, mask_panel))
    header = np.full((138, panels.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    command_color = (0, 220, 255) if navigator.direction == "RIGHT" else (255, 220, 0)
    if calibration is None:
        command_color = (0, 80, 255)

    feedback_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.92, 3)[0]
    feedback_x = max(16, (header.shape[1] - feedback_size[0]) // 2)
    cv2.putText(
        header,
        feedback,
        (feedback_x, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        command_color,
        3,
    )

    if calibration is None:
        range_line = (
            f"MEASURE CONE HEIGHT | PUT CONE {calibration_distance_cm:.0f} CM FROM CAMERA | PRESS C"
        )
    else:
        range_line = (
            f"DISTANCE: {format_distance(distance_cm)}   |   TURN: {navigator.turn_start_cm:.0f} CM   "
            f"|   HARD: {navigator.hard_turn_cm:.0f} CM   |   PASS: {navigator.pass_distance_cm:.0f} CM"
        )
    range_size = cv2.getTextSize(range_line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    range_x = max(16, (header.shape[1] - range_size[0]) // 2)
    cv2.putText(
        header,
        range_line,
        (range_x, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        2,
    )

    motion_line = (
        f"ROBOT {navigator.direction} -> CONE {navigator.expected_cone_side}   |   "
        f"PHASE: {navigator.phase}   |   PASSED: {navigator.cones_passed}   |   NEXT: {navigator.next_direction}"
    )
    motion_size = cv2.getTextSize(motion_line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    motion_x = max(16, (header.shape[1] - motion_size[0]) // 2)
    cv2.putText(
        header,
        motion_line,
        (motion_x, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 190, 190),
        1,
    )

    cv2.putText(
        header,
        "C = CALIBRATE   R = RESET SLALOM   Q = QUIT",
        (16, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (145, 145, 145),
        1,
    )
    return np.vstack((header, panels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteration 3 distance-aware cone slalom feedback")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450)
    parser.add_argument(
        "--cone-height-cm",
        type=float,
        default=30.5,
        help="measured cone height; default 30.5 cm (12 in)",
    )
    parser.add_argument(
        "--calibration-distance-cm",
        type=float,
        default=100.0,
        help="camera-to-cone distance used when C is pressed",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
    )
    parser.add_argument("--turn-start-cm", type=float, default=130.0)
    parser.add_argument("--hard-turn-cm", type=float, default=80.0)
    parser.add_argument("--pass-distance-cm", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "cone height": args.cone_height_cm,
        "calibration distance": args.calibration_distance_cm,
        "turn start distance": args.turn_start_cm,
        "hard turn distance": args.hard_turn_cm,
        "pass distance": args.pass_distance_cm,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{name} must be greater than zero")
    if not args.turn_start_cm > args.hard_turn_cm > args.pass_distance_cm:
        raise SystemExit("Distances must satisfy: turn-start > hard-turn > pass-distance")


def main() -> None:
    args = parse_args()
    validate_args(args)
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 1 and check camera permission."
        )

    calibration = CameraCalibration.load(args.calibration_file)
    navigator = DistanceAwareSlalomNavigator(
        turn_start_cm=args.turn_start_cm,
        hard_turn_cm=args.hard_turn_cm,
        pass_distance_cm=args.pass_distance_cm,
    )
    if calibration is None:
        print("Distance is not calibrated. Place the measured cone at the calibration mark and press C.")
    else:
        print(f"Loaded distance calibration from {args.calibration_file}")
    print("Iteration 3 running. C=calibrate, R=reset slalom, Q/Escape=stop.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                break

            detections = detect_cones(frame, min_area=args.min_area)
            mask = make_red_mask(frame)
            distance_cm = None
            if calibration is not None and detections:
                distance_cm = calibration.estimate_distance_cm(detections[0], frame.shape)
            feedback = navigator.update(
                detections,
                distance_cm,
                frame.shape,
                calibrated=calibration is not None,
            )
            dashboard = make_dashboard(
                frame,
                mask,
                detections,
                navigator,
                feedback,
                navigator.smoothed_distance_cm,
                calibration,
                args.calibration_distance_cm,
                display_width=args.display_width,
            )
            cv2.imshow("Cone Slalom Feedback - Iteration 3", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                navigator = DistanceAwareSlalomNavigator(
                    turn_start_cm=args.turn_start_cm,
                    hard_turn_cm=args.hard_turn_cm,
                    pass_distance_cm=args.pass_distance_cm,
                )
            if key == ord("c"):
                if not detections:
                    print("Calibration failed: no cone is detected.")
                    continue
                calibration = CameraCalibration.from_observation(
                    detections[0],
                    cone_height_cm=args.cone_height_cm,
                    distance_cm=args.calibration_distance_cm,
                    frame_shape=frame.shape,
                )
                calibration.save(args.calibration_file)
                navigator = DistanceAwareSlalomNavigator(
                    turn_start_cm=args.turn_start_cm,
                    hard_turn_cm=args.hard_turn_cm,
                    pass_distance_cm=args.pass_distance_cm,
                )
                print(
                    f"Saved calibration: {args.cone_height_cm:.1f} cm cone at "
                    f"{args.calibration_distance_cm:.1f} cm, focal length "
                    f"{calibration.focal_length_px:.1f} px"
                )
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
