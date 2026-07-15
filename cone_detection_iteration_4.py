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

    turn_start_cm: float = 160.0
    turn_start_height_ratio: float = 0.30
    visual_turn_min_bottom_ratio: float = 0.72
    hard_turn_cm: float = 80.0
    pass_distance_cm: float = 60.0
    countersteer_frames: int = 12
    pass_confirmation_frames: int = 3
    visual_turn_confirmation_frames: int = 2
    emergency_turn_min_height_ratio: float = 0.36
    emergency_turn_min_bottom_ratio: float = 0.76
    camera_offset_cm: float = 0.0
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
    close_cone_hazard: bool = False
    max_cones: int | None = None
    course_complete: bool = False
    current_target_height_ratio: float | None = None
    visual_turn_frames: int = 0
    turn_trigger_source: str | None = None
    passed_cone_side_to_ignore: str | None = None
    corrected_center_x_ratio: float | None = None
    last_tracked_target: Detection | None = None
    passed_cone_reference: Detection | None = None
    passed_cone_too_close: bool = False
    passed_cone_clearance_direction: str | None = None
    passed_cone_clear_frames: int = 0
    passed_cone_clear_confirmation_frames: int = 8

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
        self.close_cone_hazard = False
        self.current_target_height_ratio = None
        self.visual_turn_frames = 0
        self.turn_trigger_source = None
        self.corrected_center_x_ratio = None
        self.last_tracked_target = None

    def _vehicle_center_x_ratio(
        self,
        detection: Detection,
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
        distance_cm: float | None = None,
    ) -> float:
        """Express cone position relative to the vehicle, not the camera.

        ``camera_offset_cm`` is positive to the vehicle's right and negative
        to its left. A left-mounted camera sees a vehicle-centerline cone to
        the right of the image center, so its negative offset shifts that
        observation back toward a corrected ratio of 0.5.
        """
        if distance_cm is None:
            distance_cm = calibration.estimate_distance_cm(detection, frame_shape)
        if not math.isfinite(distance_cm) or distance_cm <= 0.0:
            return detection.center[0] / frame_shape[1]
        scaled_focal_px = (
            calibration.focal_length_px
            * frame_shape[0]
            / calibration.frame_height
        )
        corrected_x_px = (
            detection.center[0]
            + self.camera_offset_cm * scaled_focal_px / distance_cm
        )
        return corrected_x_px / frame_shape[1]

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

    def _is_on_passed_cone_side(
        self,
        detection: Detection,
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
    ) -> bool:
        center_ratio = self._vehicle_center_x_ratio(
            detection,
            calibration,
            frame_shape,
        )
        if self.passed_cone_side_to_ignore == "LEFT":
            return center_ratio <= 0.45
        if self.passed_cone_side_to_ignore == "RIGHT":
            return center_ratio >= 0.55
        return False

    def _match_current_target(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, ...],
    ) -> Detection | None:
        """Keep tracking the same cone instead of switching to the largest red object."""
        previous = self.current_target
        if previous is None:
            return detections[0] if detections else None

        frame_height, frame_width = frame_shape[:2]
        matches: list[tuple[float, Detection]] = []
        for detection in detections:
            x_shift = abs(detection.center[0] - previous.center[0]) / frame_width
            y_shift = abs(detection.center[1] - previous.center[1]) / frame_height
            height_scale = max(
                detection.height / max(previous.height, 1),
                previous.height / max(detection.height, 1),
            )
            width_scale = max(
                detection.width / max(previous.width, 1),
                previous.width / max(detection.width, 1),
            )
            if (
                x_shift <= 0.25
                and y_shift <= 0.35
                and height_scale <= 2.8
                and width_scale <= 2.8
            ):
                score = x_shift + 0.5 * y_shift + 0.05 * (height_scale + width_scale)
                matches.append((score, detection))
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])[1]

    def _matches_recent_passed_cone(
        self,
        detection: Detection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        """Recognize the just-passed cone despite camera motion.

        This check is used only for a visually close detection while the
        navigator is waiting for the next, necessarily distant course cone.
        Updating the reference on every match lets it follow the old cone as
        the vehicle continues its gentle counterturn.
        """
        previous = self.passed_cone_reference
        if previous is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        x_shift = abs(detection.center[0] - previous.center[0]) / frame_width
        y_shift = abs(detection.center[1] - previous.center[1]) / frame_height
        height_scale = max(
            detection.height / max(previous.height, 1),
            previous.height / max(detection.height, 1),
        )
        width_scale = max(
            detection.width / max(previous.width, 1),
            previous.width / max(detection.width, 1),
        )
        return (
            x_shift <= 0.32
            and y_shift <= 0.35
            and height_scale <= 2.2
            and width_scale <= 2.2
        )

    def _is_visually_close(
        self,
        detection: Detection,
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
        distance_cm: float | None = None,
    ) -> bool:
        if distance_cm is None:
            distance_cm = calibration.estimate_distance_cm(detection, frame_shape)
        bottom_ratio = (detection.y + detection.height) / frame_shape[0]
        height_ratio = detection.height / frame_shape[0]
        return (
            detection.cropped
            or distance_cm <= self.hard_turn_cm
            or bottom_ratio >= 0.82
            or (
                bottom_ratio >= self.visual_turn_min_bottom_ratio
                and height_ratio >= self.turn_start_height_ratio
            )
        )

    def _update_passed_cone_clearance(
        self,
        detections: list[Detection],
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
    ) -> None:
        """Hold an away-turn until the known passed cone is safely clear."""
        if self.passed_cone_reference is None:
            self.passed_cone_too_close = False
            self.passed_cone_clearance_direction = None
            self.passed_cone_clear_frames = 0
            return

        matches = [
            detection
            for detection in detections
            if self._matches_recent_passed_cone(detection, frame_shape)
        ]
        matched = min(
            matches,
            key=lambda item: (
                abs(item.center[0] - self.passed_cone_reference.center[0])
                + abs(item.center[1] - self.passed_cone_reference.center[1])
            ),
            default=None,
        )
        if matched is not None:
            self.passed_cone_reference = matched
            distance = calibration.estimate_distance_cm(matched, frame_shape)
            if self._is_visually_close(
                matched,
                calibration,
                frame_shape,
                distance,
            ):
                center_ratio = self._vehicle_center_x_ratio(
                    matched,
                    calibration,
                    frame_shape,
                    distance,
                )
                self.passed_cone_too_close = True
                self.passed_cone_clearance_direction = (
                    "RIGHT" if center_ratio < 0.5 else "LEFT"
                )
                self.passed_cone_clear_frames = 0
                return

        self.passed_cone_clear_frames += 1
        if self.passed_cone_clear_frames >= self.passed_cone_clear_confirmation_frames:
            self.passed_cone_reference = None
            self.passed_cone_too_close = False
            self.passed_cone_clearance_direction = None
            self.passed_cone_clear_frames = 0

    def _select_new_target(
        self,
        detections: list[Detection],
        calibration: CameraCalibration,
        frame_shape: tuple[int, ...],
    ) -> Detection | None:
        # A close, ambiguous cone is a latched safety stop. A dropped camera
        # frame must not restart the motors; only resetting the navigator
        # clears this state.
        if self.close_cone_hazard:
            return None
        if not detections:
            return None
        if not self.awaiting_new_cone:
            # Once a next cone is accepted, follow its frame-to-frame motion
            # even if camera yaw carries it across the old cone's image half.
            return self._match_current_target(detections, frame_shape)

        # The passed cone may remain in view while the camera countersteers.
        # A new course cone should be farther away, not clipped by the image
        # bottom, and generally closer to the forward center of the image.
        candidates: list[Detection] = []
        for detection in detections:
            distance = calibration.estimate_distance_cm(detection, frame_shape)
            bottom_ratio = (detection.y + detection.height) / frame_shape[0]
            height_ratio = detection.height / frame_shape[0]
            center_ratio = self._vehicle_center_x_ratio(
                detection,
                calibration,
                frame_shape,
                distance,
            )
            visually_close = self._is_visually_close(
                detection,
                calibration,
                frame_shape,
                distance,
            )
            if visually_close and self._matches_recent_passed_cone(
                detection,
                frame_shape,
            ):
                # This is the known cone that was just passed, not a new
                # obstacle. Continue tracking its motion but never reacquire it.
                self.passed_cone_reference = detection
                continue
            if self._is_on_passed_cone_side(
                detection,
                calibration,
                frame_shape,
            ):
                continue
            if visually_close:
                if 0.20 <= center_ratio <= 0.80:
                    # A close centered cone may be the old cone or an
                    # unexpected obstacle, so latch a STOP for operator review.
                    self.close_cone_hazard = True
                # A close edge cone is normally the cone just passed. Never
                # accept either case as the next far course target.
                continue
            if (
                not detection.cropped
                and distance >= self.pass_distance_cm * 1.35
                and bottom_ratio < 0.86
            ):
                candidates.append(detection)
        if self.close_cone_hazard:
            return None
        if not candidates:
            return None

        target = min(
            candidates,
            key=lambda item: (
                abs(
                    self._vehicle_center_x_ratio(
                        item,
                        calibration,
                        frame_shape,
                    )
                    - 0.5
                ),
                -(item.width * item.height),
            ),
        )
        self.awaiting_new_cone = False
        return target

    def _finish_pass(self) -> None:
        self.passed_cone_reference = (
            self.current_target or self.last_tracked_target
        )
        self.passed_cone_too_close = self.passed_cone_reference is not None
        self.passed_cone_clear_frames = 0
        if self.corrected_center_x_ratio is not None:
            self.passed_cone_clearance_direction = (
                "RIGHT" if self.corrected_center_x_ratio < 0.5 else "LEFT"
            )
        self.cones_passed += 1
        if self.max_cones is not None and self.cones_passed >= self.max_cones:
            self.course_complete = True
            self.phase = "COMPLETE"
            self.countersteer_remaining = 0
            self.awaiting_new_cone = False
            self.passed_cone_side_to_ignore = None
            self.passed_cone_reference = None
            self.passed_cone_too_close = False
            self.passed_cone_clearance_direction = None
            self._reset_target_tracking()
            return
        self.direction_index = 1 - self.direction_index
        # After a RIGHT pass the old cone is on the left and the new direction
        # is LEFT (and vice versa), so the new direction names the old side.
        self.passed_cone_side_to_ignore = self.direction
        self.phase = "COUNTERSTEERING"
        self.countersteer_remaining = self.countersteer_frames
        self.awaiting_new_cone = True
        self._reset_target_tracking()

    def update(
        self,
        detections: list[Detection],
        calibration: CameraCalibration | None,
        frame_shape: tuple[int, ...],
        motion_applied: bool = True,
    ) -> str:
        if self.course_complete:
            return self.feedback(calibrated=calibration is not None, frame_shape=frame_shape)
        if calibration is None:
            self.phase = "CALIBRATION REQUIRED"
            self.current_target = None
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"

        self._update_passed_cone_clearance(
            detections,
            calibration,
            frame_shape,
        )

        # Immediately after a pass, countersteer away from the old cone. Keep
        # examining frames during this interval: a safe, distant next cone can
        # end the blind countersteer early, and a close centered cone must stop
        # the robot instead of being ignored.
        countersteer_target: Detection | None = None
        if self.countersteer_remaining > 0:
            # Early acquisition is deliberately limited to the forward 60%
            # of the image. A far version of the cone just passed can remain
            # at an outer edge and must not be mistaken for the next cone.
            forward_detections = [
                detection
                for detection in detections
                if 0.20
                <= self._vehicle_center_x_ratio(
                    detection,
                    calibration,
                    frame_shape,
                )
                <= 0.80
            ]
            countersteer_target = self._select_new_target(
                forward_detections,
                calibration,
                frame_shape,
            )
            if self.close_cone_hazard:
                self.countersteer_remaining = 0
                self.phase = "SEARCHING"
                self.current_target = None
                self.raw_distance_cm = None
                self.smoothed_distance_cm = None
                return self.feedback(calibrated=True, frame_shape=frame_shape)
            if countersteer_target is not None:
                self.countersteer_remaining = 0
                self.phase = "SEARCHING"
            else:
                if motion_applied:
                    self.countersteer_remaining -= 1
                self.current_target = None
                self.raw_distance_cm = None
                self.smoothed_distance_cm = None
                if self.countersteer_remaining > 0:
                    return self.feedback(calibrated=True, frame_shape=frame_shape)
                self.phase = "SEARCHING"

        target = countersteer_target or self._select_new_target(
            detections,
            calibration,
            frame_shape,
        )
        self.current_target = target
        if target is None:
            self.raw_distance_cm = None
            self.current_target_height_ratio = None
            if self.phase == "APPROACHING":
                self.visual_turn_frames = 0
            if self.pass_armed:
                self.missing_frames += 1
                if self.missing_frames >= self.pass_confirmation_frames:
                    self._finish_pass()
            elif self.phase == "APPROACHING":
                self.phase = "SEARCHING"
            return self.feedback(calibrated=True, frame_shape=frame_shape)

        self.last_tracked_target = target
        raw_distance = calibration.estimate_distance_cm(target, frame_shape)
        if target.cropped:
            # Pixel height underestimates the full cone after frame clipping.
            # Treat it as conservatively close instead of delaying the turn.
            raw_distance = min(raw_distance, self.hard_turn_cm)
        self.raw_distance_cm = raw_distance
        center_x_ratio = self._vehicle_center_x_ratio(
            target,
            calibration,
            frame_shape,
            raw_distance,
        )
        self.corrected_center_x_ratio = center_x_ratio
        height_ratio = target.height / frame_shape[0]
        self.current_target_height_ratio = height_ratio
        bottom_ratio = (target.y + target.height) / frame_shape[0]

        if self.phase in {"SEARCHING", "CALIBRATION REQUIRED"}:
            self.phase = "APPROACHING"
        if self.smoothed_distance_cm is None:
            self.smoothed_distance_cm = raw_distance
        else:
            self.smoothed_distance_cm = 0.68 * self.smoothed_distance_cm + 0.32 * raw_distance

        self.closest_distance_cm = min(self.closest_distance_cm, raw_distance)
        self.tallest_target_px = max(self.tallest_target_px, target.height)

        if self.phase == "APPROACHING":
            # Calibrated range remains the normal turn trigger. If that
            # calibration is badly wrong, do not continue driving into a cone
            # that is unmistakably large and low in the camera view. Requiring
            # two consecutive frames and stricter visual limits than the
            # general close-cone safety check avoids the former early turns.
            visual_turn_candidate = (
                bottom_ratio >= self.emergency_turn_min_bottom_ratio
                and height_ratio >= self.emergency_turn_min_height_ratio
            )
            if visual_turn_candidate:
                self.visual_turn_frames += 1
            else:
                self.visual_turn_frames = 0

            distance_turn_ready = (
                raw_distance <= self.turn_start_cm
                or self.smoothed_distance_cm <= self.turn_start_cm
            )
            visual_turn_ready = (
                self.visual_turn_frames >= self.visual_turn_confirmation_frames
            )
            if distance_turn_ready or visual_turn_ready:
                self.phase = "TURNING"
                self.turn_start_x_ratio = center_x_ratio
                self.turn_trigger_source = (
                    "DISTANCE" if distance_turn_ready else "CLOSE CONE BACKUP"
                )

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
        if self.course_complete:
            return f"COURSE COMPLETE - {self.cones_passed}/{self.max_cones} CONES - STOP"
        if not calibrated:
            return "CALIBRATE DISTANCE: PLACE CONE AT MARK + PRESS C"
        if self.passed_cone_too_close and self.passed_cone_clearance_direction:
            return (
                f"CLEAR {self.passed_cone_clearance_direction} - "
                "PASSED CONE STILL CLOSE"
            )
        if self.countersteer_remaining > 0 or self.phase == "COUNTERSTEERING":
            return f"TURN {self.direction} NOW - CONE {self.cones_passed} PASSED"
        if self.current_target is None:
            if self.close_cone_hazard:
                return "STOP - CLOSE CONE DETECTED WHILE SEARCHING"
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
        center_x_ratio = (
            self.corrected_center_x_ratio
            if self.corrected_center_x_ratio is not None
            else self.current_target.center[0] / frame_shape[1]
        )

        if self.pass_armed:
            return f"HOLD {self.direction} - CLEARING CONE AT {distance_text}"
        if self.phase not in {"TURNING", "PASSING"} and distance > self.turn_start_cm:
            return (
                f"FORWARD - CONE {distance_text} - "
                f"TURN AT {self.turn_start_cm:.0f} cm"
            )

        cleared_side = self._has_cleared_to_expected_side(center_x_ratio)
        if cleared_side:
            return f"HOLD {self.direction} - CONE SLID {self.expected_cone_side}"
        if self.turn_trigger_source == "CLOSE CONE BACKUP":
            return f"BEGIN SLOW {self.direction} TURN - CLOSE CONE VISUAL BACKUP"
        return f"BEGIN SLOW {self.direction} TURN - TRACK CONE MOTION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteration 4 camera-motion-aware cone slalom feedback"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450)
    parser.add_argument(
        "--robot-width-cm",
        type=float,
        default=30.48,
        help="vehicle width; 30.48 cm is 12 inches",
    )
    parser.add_argument(
        "--camera-from-left-cm",
        type=float,
        default=7.62,
        help="camera center measured from the vehicle's left side; 7.62 cm is 3 inches",
    )
    parser.add_argument("--cone-height-cm", type=float, default=30.5)
    parser.add_argument("--calibration-distance-cm", type=float, default=100.0)
    parser.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE
    )
    parser.add_argument("--turn-start-cm", type=float, default=160.0)
    parser.add_argument(
        "--turn-start-height-ratio",
        type=float,
        default=0.30,
        help="close-cone safety threshold as a fraction of image height",
    )
    parser.add_argument(
        "--hard-turn-cm",
        type=float,
        default=80.0,
        help=(
            "legacy option name for the close-cone clearance threshold; "
            "it never increases motor speed"
        ),
    )
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
        args.robot_width_cm,
    ) <= 0:
        raise SystemExit("All physical measurements must be greater than zero")
    if not args.turn_start_cm > args.hard_turn_cm > args.pass_distance_cm:
        raise SystemExit("Distances must satisfy: turn-start > hard-turn > pass-distance")
    if not 0.0 < args.turn_start_height_ratio < 1.0:
        raise SystemExit("turn-start-height-ratio must be between 0 and 1")
    if args.countersteer_frames < 1:
        raise SystemExit("countersteer-frames must be at least 1")
    if not 0.0 <= args.camera_from_left_cm <= args.robot_width_cm:
        raise SystemExit("camera-from-left must be between zero and robot-width")


def new_navigator(args: argparse.Namespace) -> CameraMotionSlalomNavigator:
    return CameraMotionSlalomNavigator(
        turn_start_cm=args.turn_start_cm,
        turn_start_height_ratio=args.turn_start_height_ratio,
        hard_turn_cm=args.hard_turn_cm,
        pass_distance_cm=args.pass_distance_cm,
        countersteer_frames=args.countersteer_frames,
        camera_offset_cm=(
            args.camera_from_left_cm - args.robot_width_cm / 2.0
        ),
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
                if all_detections[0].cropped:
                    print("Calibration failed: keep the full cone inside the image.")
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
