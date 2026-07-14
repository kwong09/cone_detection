"""Camera-motion-aware cone slalom feedback — Iteration 4.

Iteration 3 assumed cone position could be judged against fixed image zones.
That breaks when the camera is mounted to a turning robot. This version records
the cone's position when a turn begins, measures its motion relative to that
starting point, and recognizes a pass when a correctly cleared close cone
starts moving away or leaves the image. It then immediately countersteers and
temporarily ignores the old cone while looking for the next head-on target.

This program provides steering feedback only. It does not command motors yet.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from cone_detection_iteration_1 import Detection, detect_cones, make_red_mask
from cone_detection_iteration_3 import (
    DEFAULT_CALIBRATION_FILE,
    CameraCalibration,
    format_distance,
    make_dashboard,
)


@dataclass
class CameraMotionSlalomNavigator:
    """Time a slalom pass using distance and cone motion in a turning camera."""

    turn_start_cm: float = 130.0
    hard_turn_cm: float = 80.0
    pass_distance_cm: float = 60.0
    countersteer_frames: int = 12
    pass_confirmation_frames: int = 3
    direction_index: int = 0
    cones_passed: int = 0
    phase: str = "SEARCHING"
    current_target: Detection | None = None
    raw_distance_cm: float | None = None
    smoothed_distance_cm: float | None = None
    turn_start_x_ratio: float | None = None
    closest_distance_cm: float = math.inf
    tallest_target_px: int = 0
    clearance_seen: bool = False
    pass_armed: bool = False
    moving_away_frames: int = 0
    missing_frames: int = 0
    countersteer_remaining: int = 0
    awaiting_new_cone: bool = False

    @property
    def direction(self) -> str:
        return ("RIGHT", "LEFT")[self.direction_index]

    @property
    def next_direction(self) -> str:
        return ("LEFT", "RIGHT")[self.direction_index]

    @property
    def expected_cone_side(self) -> str:
        return "LEFT" if self.direction == "RIGHT" else "RIGHT"

    def _reset_target_tracking(self) -> None:
        self.current_target = None
        self.raw_distance_cm = None
        self.smoothed_distance_cm = None
        self.turn_start_x_ratio = None
        self.closest_distance_cm = math.inf
        self.tallest_target_px = 0
        self.clearance_seen = False
        self.pass_armed = False
        self.moving_away_frames = 0
        self.missing_frames = 0

    def _relative_side_progress(self, center_x_ratio: float) -> float:
        if self.turn_start_x_ratio is None:
            return 0.0
        if self.direction == "RIGHT":
            return self.turn_start_x_ratio - center_x_ratio
        return center_x_ratio - self.turn_start_x_ratio

    def _has_cleared_to_expected_side(self, center_x_ratio: float) -> bool:
        # Relative motion handles camera yaw. The absolute half-frame check is a
        # backup when the cone was already off-center as the turn began.
        moved_across_image = self._relative_side_progress(center_x_ratio) >= 0.10
        in_expected_half = (
            center_x_ratio <= 0.42
            if self.direction == "RIGHT"
            else center_x_ratio >= 0.58
        )
        return moved_across_image or in_expected_half

    def _select_new_target(
        self,
        detections: list[Detection],
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
    ) -> Detection | None:
        if not detections:
            return None
        if not self.awaiting_new_cone:
            return detections[0]

        # The passed cone may remain in view while the camera countersteers.
        # A new course cone should be farther away, not clipped by the image
        # bottom, and generally closer to the forward center of the image.
        candidates: list[Detection] = []
        for detection in detections:
            distance = calibration.estimate_distance_cm(detection, frame_shape)
            bottom_ratio = (detection.y + detection.height) / frame_shape[0]
            if distance >= self.pass_distance_cm * 1.35 and bottom_ratio < 0.86:
                candidates.append(detection)
        if not candidates:
            return None

        target = min(
            candidates,
            key=lambda item: (
                abs(item.center[0] / frame_shape[1] - 0.5),
                -(item.width * item.height),
            ),
        )
        self.awaiting_new_cone = False
        return target

    def _finish_pass(self) -> None:
        self.cones_passed += 1
        self.direction_index = 1 - self.direction_index
        self.phase = "COUNTERSTEERING"
        self.countersteer_remaining = self.countersteer_frames
        self.awaiting_new_cone = True
        self._reset_target_tracking()

    def update(
        self,
        detections: list[Detection],
        calibration: CameraCalibration | None,
        frame_shape: tuple[int, ...],
    ) -> str:
        if calibration is None:
            self.phase = "CALIBRATION REQUIRED"
            self.current_target = None
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"

        # Immediately after a pass, command the opposite turn and ignore the
        # old cone while the robot-mounted camera rotates toward the next gate.
        if self.countersteer_remaining > 0:
            self.countersteer_remaining -= 1
            self.current_target = None
            self.raw_distance_cm = None
            self.smoothed_distance_cm = None
            if self.countersteer_remaining == 0:
                self.phase = "SEARCHING"
            return self.feedback(calibrated=True, frame_shape=frame_shape)

        target = self._select_new_target(detections, calibration, frame_shape)
        self.current_target = target
        if target is None:
            self.raw_distance_cm = None
            if self.pass_armed:
                self.missing_frames += 1
                if self.missing_frames >= self.pass_confirmation_frames:
                    self._finish_pass()
            elif self.phase == "APPROACHING":
                self.phase = "SEARCHING"
            return self.feedback(calibrated=True, frame_shape=frame_shape)

        raw_distance = calibration.estimate_distance_cm(target, frame_shape)
        self.raw_distance_cm = raw_distance
        center_x_ratio = target.center[0] / frame_shape[1]
        bottom_ratio = (target.y + target.height) / frame_shape[0]

        if self.phase in {"SEARCHING", "CALIBRATION REQUIRED"}:
            self.phase = "APPROACHING"
        if self.smoothed_distance_cm is None:
            self.smoothed_distance_cm = raw_distance
        else:
            self.smoothed_distance_cm = 0.68 * self.smoothed_distance_cm + 0.32 * raw_distance

        self.closest_distance_cm = min(self.closest_distance_cm, raw_distance)
        self.tallest_target_px = max(self.tallest_target_px, target.height)

        if (
            self.phase == "APPROACHING"
            and self.smoothed_distance_cm <= self.turn_start_cm
        ):
            self.phase = "TURNING"
            self.turn_start_x_ratio = center_x_ratio

        if self.phase in {"TURNING", "PASSING"}:
            cleared_side = self._has_cleared_to_expected_side(center_x_ratio)
            visually_close = (
                raw_distance <= self.hard_turn_cm or bottom_ratio >= 0.82
            )
            if cleared_side and visually_close:
                self.clearance_seen = True

            # Use both calibrated range and the cone approaching the bottom of
            # the frame. Camera yaw and perspective can make either cue noisy.
            close_enough_to_pass = (
                raw_distance <= self.pass_distance_cm * 1.15
                or bottom_ratio >= 0.89
            )
            if self.clearance_seen and close_enough_to_pass:
                self.pass_armed = True
                self.phase = "PASSING"

        if self.pass_armed:
            cleared_side = self._has_cleared_to_expected_side(center_x_ratio)
            moving_away_by_range = (
                raw_distance >= self.closest_distance_cm * 1.18
            )
            moving_away_by_size = (
                self.tallest_target_px > 0
                and target.height <= self.tallest_target_px * 0.82
            )
            exited_side_edge = (
                center_x_ratio <= 0.06
                if self.direction == "RIGHT"
                else center_x_ratio >= 0.94
            )
            passed_this_frame = (
                exited_side_edge
                or (cleared_side and (moving_away_by_range or moving_away_by_size))
            )
            self.moving_away_frames = (
                self.moving_away_frames + 1 if passed_this_frame else 0
            )
            self.missing_frames = 0
            if self.moving_away_frames >= self.pass_confirmation_frames:
                self._finish_pass()

        return self.feedback(calibrated=True, frame_shape=frame_shape)

    def feedback(self, calibrated: bool, frame_shape: tuple[int, ...]) -> str:
        if not calibrated:
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"
        if self.countersteer_remaining > 0 or self.phase == "COUNTERSTEERING":
            return f"TURN {self.direction} NOW - CONE {self.cones_passed} PASSED"
        if self.current_target is None:
            if self.pass_armed:
                return f"HOLD {self.direction} - CONE LEAVING VIEW"
            if self.awaiting_new_cone:
                return f"TURN {self.direction} - FIND NEXT CONE"
            if self.phase == "TURNING":
                return "STOP - CONE LOST BEFORE CLEARANCE"
            return "STOP - FIND NEXT CONE"

        distance = self.smoothed_distance_cm
        if distance is None:
            return "STOP - DISTANCE UNAVAILABLE"
        distance_text = format_distance(distance)
        center_x_ratio = self.current_target.center[0] / frame_shape[1]

        if self.pass_armed:
            return f"HOLD {self.direction} - CLEARING CONE AT {distance_text}"
        if distance > self.turn_start_cm:
            if center_x_ratio < 0.44:
                return f"ALIGN LEFT - CONE {distance_text}"
            if center_x_ratio > 0.56:
                return f"ALIGN RIGHT - CONE {distance_text}"
            return f"FORWARD - CONE {distance_text}"

        cleared_side = self._has_cleared_to_expected_side(center_x_ratio)
        if distance <= self.hard_turn_cm and not cleared_side:
            return f"HARD {self.direction} - CONE {distance_text}"
        if cleared_side:
            return f"HOLD {self.direction} - CONE SLID {self.expected_cone_side}"
        return f"BEGIN {self.direction} TURN - TRACK CONE MOTION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteration 4 camera-motion-aware cone slalom feedback"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450)
    parser.add_argument("--cone-height-cm", type=float, default=30.5)
    parser.add_argument("--calibration-distance-cm", type=float, default=100.0)
    parser.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE
    )
    parser.add_argument("--turn-start-cm", type=float, default=130.0)
    parser.add_argument("--hard-turn-cm", type=float, default=80.0)
    parser.add_argument("--pass-distance-cm", type=float, default=60.0)
    parser.add_argument("--countersteer-frames", type=int, default=12)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(
        args.cone_height_cm,
        args.calibration_distance_cm,
        args.turn_start_cm,
        args.hard_turn_cm,
        args.pass_distance_cm,
    ) <= 0:
        raise SystemExit("All physical measurements must be greater than zero")
    if not args.turn_start_cm > args.hard_turn_cm > args.pass_distance_cm:
        raise SystemExit("Distances must satisfy: turn-start > hard-turn > pass-distance")
    if args.countersteer_frames < 1:
        raise SystemExit("countersteer-frames must be at least 1")


def new_navigator(args: argparse.Namespace) -> CameraMotionSlalomNavigator:
    return CameraMotionSlalomNavigator(
        turn_start_cm=args.turn_start_cm,
        hard_turn_cm=args.hard_turn_cm,
        pass_distance_cm=args.pass_distance_cm,
        countersteer_frames=args.countersteer_frames,
    )


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
    navigator = new_navigator(args)
    if calibration is None:
        print("Distance is not calibrated. Place the measured cone at the mark and press C.")
    else:
        print(f"Loaded distance calibration from {args.calibration_file}")
    print("Iteration 4 running. C=calibrate, R=reset slalom, Q/Escape=stop.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                break

            all_detections = detect_cones(frame, min_area=args.min_area)
            mask = make_red_mask(frame)
            feedback = navigator.update(all_detections, calibration, frame.shape)
            target_detections = (
                [navigator.current_target] if navigator.current_target is not None else []
            )
            dashboard = make_dashboard(
                frame,
                mask,
                target_detections,
                navigator,
                feedback,
                navigator.smoothed_distance_cm,
                calibration,
                args.calibration_distance_cm,
                display_width=args.display_width,
            )
            cv2.imshow("Cone Slalom Feedback - Iteration 4", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                navigator = new_navigator(args)
            if key == ord("c"):
                if not all_detections:
                    print("Calibration failed: no cone is detected.")
                    continue
                calibration = CameraCalibration.from_observation(
                    all_detections[0],
                    cone_height_cm=args.cone_height_cm,
                    distance_cm=args.calibration_distance_cm,
                    frame_shape=frame.shape,
                )
                calibration.save(args.calibration_file)
                navigator = new_navigator(args)
                print(
                    f"Saved calibration: {args.cone_height_cm:.1f} cm cone at "
                    f"{args.calibration_distance_cm:.1f} cm"
                )
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
