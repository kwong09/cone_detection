"""Cone Detection and Head-On Slalom Feedback — Iteration 2.

This version reuses Iteration 1's tuned red-cone detector. Its navigation state
machine is designed for approaching each cone head-on: steering right should
make the cone travel toward the left of the camera, while steering left should
make it travel toward the right. A pass is counted only after a close cone has
reached the expected side and then left the camera view.

This program provides steering feedback only. It does not command motors yet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from cone_detection_iteration_1 import (
    Detection,
    detect_cones,
    draw_detections,
    draw_panel_title,
    make_red_mask,
)


@dataclass
class HeadOnSlalomNavigator:
    """Track head-on cone approaches and alternate the passing side."""

    direction_index: int = 0
    cones_passed: int = 0
    saw_target: bool = False
    saw_close_cone: bool = False
    reached_pass_side: bool = False
    candidate_frames: int = 0
    largest_target_area: int = 0
    pass_confirmation_frames: int = 5

    @property
    def direction(self) -> str:
        return ("RIGHT", "LEFT")[self.direction_index]

    @property
    def next_direction(self) -> str:
        return ("LEFT", "RIGHT")[self.direction_index]

    @property
    def expected_cone_side(self) -> str:
        # Camera motion is opposite robot motion.
        return "LEFT" if self.direction == "RIGHT" else "RIGHT"

    def _begin_next_cone(self) -> None:
        self.cones_passed += 1
        self.direction_index = 1 - self.direction_index
        self.saw_target = False
        self.saw_close_cone = False
        self.reached_pass_side = False
        self.candidate_frames = 0
        self.largest_target_area = 0

    def update(self, detections: list[Detection], frame_shape: tuple[int, ...]) -> str:
        """Update the head-on pass state and return the steering feedback."""
        frame_height, frame_width = frame_shape[:2]
        target = detections[0] if detections else None

        if target is not None:
            center_x = target.center[0] / frame_width
            bottom = (target.y + target.height) / frame_height
            area = target.width * target.height
            area_ratio = area / (frame_width * frame_height)
            is_close = bottom >= 0.72 or area_ratio >= 0.035
            is_on_pass_side = (
                center_x <= 0.38 if self.direction == "RIGHT" else center_x >= 0.62
            )

            self.saw_target = True
            self.largest_target_area = max(self.largest_target_area, area)
            if is_close:
                self.saw_close_cone = True
            if is_close and is_on_pass_side:
                self.reached_pass_side = True

            # Once the close cone has reached the correct edge, a much smaller
            # detection is probably the next cone farther down the course.
            new_far_target = (
                self.reached_pass_side
                and area < self.largest_target_area * 0.42
                and not is_close
            )
            self.candidate_frames = self.candidate_frames + 1 if new_far_target else 0
        elif self.reached_pass_side:
            self.candidate_frames += 1
        else:
            # Losing a centered cone is not proof of a pass. Keep the current
            # direction until it has visibly moved to the expected edge.
            self.candidate_frames = 0

        if self.reached_pass_side and self.candidate_frames >= self.pass_confirmation_frames:
            self._begin_next_cone()

        return self.feedback(detections, frame_shape)

    def feedback(self, detections: list[Detection], frame_shape: tuple[int, ...]) -> str:
        """Return a specific instruction for the current approach phase."""
        if not detections:
            if self.reached_pass_side:
                return f"HOLD {self.direction} - FINISH PASS"
            if self.saw_close_cone:
                return f"CONE LOST - HOLD {self.direction}"
            return f"FIND CONE - BIAS {self.direction}"

        target = detections[0]
        frame_height, frame_width = frame_shape[:2]
        center_x = target.center[0] / frame_width
        bottom = (target.y + target.height) / frame_height
        area_ratio = (target.width * target.height) / (frame_width * frame_height)
        is_close = bottom >= 0.72 or area_ratio >= 0.035
        is_on_pass_side = center_x <= 0.38 if self.direction == "RIGHT" else center_x >= 0.62

        if self.reached_pass_side:
            return f"HOLD {self.direction} - PASSING CONE"
        if is_close and not is_on_pass_side:
            return f"HARD {self.direction} - CLEAR THE CONE"
        if is_on_pass_side:
            return f"KEEP {self.direction} - CONE MOVING {self.expected_cone_side}"
        return f"STEER {self.direction} - SEND CONE {self.expected_cone_side}"

    def phase_text(self, detections: list[Detection]) -> str:
        if self.reached_pass_side:
            return "PASS ARMED"
        if self.saw_close_cone:
            return "CLOSE - MOVE CONE TO EDGE"
        if detections:
            return "HEAD-ON APPROACH"
        return "SEARCHING"


def make_dashboard(
    frame: np.ndarray,
    mask: np.ndarray,
    detections: list[Detection],
    navigator: HeadOnSlalomNavigator,
    feedback: str,
    display_width: int = 1600,
) -> np.ndarray:
    """Render the live camera, mask, and head-on slalom guidance."""
    camera_panel = draw_detections(frame, detections)
    mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    panel_width = max(320, display_width // 2)
    panel_height = max(180, round(frame.shape[0] * panel_width / frame.shape[1]))
    panel_size = (panel_width, panel_height)
    camera_panel = cv2.resize(camera_panel, panel_size, interpolation=cv2.INTER_AREA)
    mask_panel = cv2.resize(mask_panel, panel_size, interpolation=cv2.INTER_NEAREST)
    draw_panel_title(camera_panel, "CAMERA + DETECTIONS")
    draw_panel_title(mask_panel, "RED CONE MASK")

    panels = np.hstack((camera_panel, mask_panel))
    header = np.full((116, panels.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    command_color = (0, 220, 255) if navigator.direction == "RIGHT" else (255, 220, 0)
    text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    feedback_x = max(16, (header.shape[1] - text_size[0]) // 2)
    cv2.putText(
        header,
        feedback,
        (feedback_x, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        command_color,
        3,
    )

    expected_motion = (
        f"ROBOT {navigator.direction}  ->  CONE SHOULD MOVE {navigator.expected_cone_side}"
    )
    motion_size = cv2.getTextSize(expected_motion, cv2.FONT_HERSHEY_SIMPLEX, 0.61, 2)[0]
    motion_x = max(16, (header.shape[1] - motion_size[0]) // 2)
    cv2.putText(
        header,
        expected_motion,
        (motion_x, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (235, 235, 235),
        2,
    )

    status = (
        f"CONE {navigator.cones_passed + 1}: {navigator.phase_text(detections)}   |   "
        f"PASSED: {navigator.cones_passed}   |   NEXT: {navigator.next_direction}"
    )
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    status_x = max(16, (header.shape[1] - status_size[0]) // 2)
    cv2.putText(
        header,
        status,
        (status_x, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 190, 190),
        1,
    )
    return np.vstack((header, panels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteration 2 head-on red-cone slalom feedback")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450, help="ignore smaller red regions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 1 and check camera permission."
        )

    navigator = HeadOnSlalomNavigator()
    print("Iteration 2 running. Press R to reset; Q or Escape to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                break

            detections = detect_cones(frame, min_area=args.min_area)
            mask = make_red_mask(frame)
            feedback = navigator.update(detections, frame.shape)
            dashboard = make_dashboard(
                frame,
                mask,
                detections,
                navigator,
                feedback,
                display_width=args.display_width,
            )
            cv2.imshow("Cone Slalom Feedback - Iteration 2", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                navigator = HeadOnSlalomNavigator()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
