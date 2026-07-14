"""Cone Detection — Iteration 1.

This version detects the red cones reliably in the current room lighting and
shows the camera and mask side by side. Its slalom feedback is experimental:
it only infers that a cone was passed after a nearby detection disappears, so
it does not understand the cone's expected side-to-side motion during a pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass
class SlalomNavigator:
    """Alternate steering direction after each nearby cone is passed."""

    direction_index: int = 0
    cones_passed: int = 0
    saw_close_cone: bool = False
    missing_frames: int = 0
    last_close_area: int = 0
    pass_confirmation_frames: int = 6

    @property
    def direction(self) -> str:
        return ("RIGHT", "LEFT")[self.direction_index]

    @property
    def next_direction(self) -> str:
        return ("LEFT", "RIGHT")[self.direction_index]

    def update(self, detections: list[Detection], frame_shape: tuple[int, ...]) -> str:
        """Update navigation state and return the current steering direction.

        A cone must first be close to the bottom of the image. It is considered
        passed only after it disappears, or is replaced by a much smaller cone,
        for several consecutive frames. This prevents a brief missed detection
        from reversing the steering command.
        """
        frame_height, frame_width = frame_shape[:2]
        nearest = detections[0] if detections else None

        if nearest is not None:
            area = nearest.width * nearest.height
            bottom_ratio = (nearest.y + nearest.height) / frame_height
            area_ratio = area / (frame_width * frame_height)
            is_close = bottom_ratio >= 0.72 or area_ratio >= 0.035

            if is_close:
                self.saw_close_cone = True
                self.last_close_area = max(self.last_close_area, area)
                self.missing_frames = 0
            elif self.saw_close_cone and area < self.last_close_area * 0.45:
                # The nearby cone left the frame while a farther cone remains.
                self.missing_frames += 1
            else:
                self.missing_frames = 0
        elif self.saw_close_cone:
            self.missing_frames += 1
        else:
            self.missing_frames = 0

        if self.saw_close_cone and self.missing_frames >= self.pass_confirmation_frames:
            self.cones_passed += 1
            self.direction_index = 1 - self.direction_index
            self.saw_close_cone = False
            self.missing_frames = 0
            self.last_close_area = 0

        return self.direction

    def tracking_text(self, detections: list[Detection]) -> str:
        if self.saw_close_cone:
            return f"PASSING CONE {self.cones_passed + 1}"
        if detections:
            return f"APPROACHING CONE {self.cones_passed + 1}"
        return f"SEARCHING FOR CONE {self.cones_passed + 1}"


def make_red_mask(frame: np.ndarray) -> np.ndarray:
    """Return a mask tuned for the saturated red cones in the test room."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around OpenCV's 0/179 hue boundary. These deliberately narrow
    # ranges were measured from the supplied room screenshot. Skin was around
    # hue 4-10, the wood around 11-13, and the red ball around 170-175.
    low_red = cv2.inRange(hsv, (0, 140, 65), (2, 255, 255))
    high_red = cv2.inRange(hsv, (176, 140, 65), (179, 255, 255))
    mask = cv2.bitwise_or(low_red, high_red)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2
    )


def contour_taper_ratio(contour: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    """Compare a contour's lower width to its upper width."""
    silhouette = np.zeros((height, width), dtype=np.uint8)
    shifted = contour - np.array([[[x, y]]])
    cv2.drawContours(silhouette, [shifted], -1, 255, -1)

    row_widths = np.count_nonzero(silhouette, axis=1)
    third = max(1, height // 3)
    top_widths = row_widths[:third]
    bottom_widths = row_widths[-third:]
    top_widths = top_widths[top_widths > 0]
    bottom_widths = bottom_widths[bottom_widths > 0]
    if len(top_widths) == 0 or len(bottom_widths) == 0:
        return 0.0
    return float(np.median(bottom_widths) / np.median(top_widths))


def detect_cones(
    frame: np.ndarray, min_area: int = 450, min_confidence: float = 0.55
) -> list[Detection]:
    """Detect saturated red regions that also have a cone silhouette."""
    mask = make_red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        # Reject balls, broad red objects, and horizontal patches.
        if width == 0 or height < width * 1.05:
            continue

        fill_ratio = area / (width * height)
        aspect_ratio = height / width
        taper_ratio = contour_taper_ratio(contour, x, y, width, height)

        # A cone should be substantially wider in its bottom third than in its
        # top third. This is the main shape check that prevents arbitrary red
        # rectangles or narrow vertical regions from being called cones.
        if not 0.20 <= fill_ratio <= 0.82 or taper_ratio < 1.35:
            continue

        fill_score = max(0.0, 1.0 - abs(fill_ratio - 0.52) / 0.40)
        aspect_score = min(1.0, aspect_ratio / 1.35)
        taper_score = min(1.0, (taper_ratio - 1.0) / 1.25)
        confidence = 0.40 * fill_score + 0.25 * aspect_score + 0.35 * taper_score

        if confidence >= min_confidence:
            detections.append(Detection(x, y, width, height, confidence))

    return sorted(detections, key=lambda item: item.width * item.height, reverse=True)


def horizontal_position(center_x: int, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "LEFT"
    if center_x > frame_width * 2 / 3:
        return "RIGHT"
    return "CENTER"


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    cv2.line(output, (frame_width // 3, 0), (frame_width // 3, frame_height), (80, 80, 80), 1)
    cv2.line(output, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (80, 80, 80), 1)

    for detection in detections:
        center_x, center_y = detection.center
        position = horizontal_position(center_x, frame_width)
        cv2.rectangle(
            output,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (0, 255, 0),
            2,
        )
        cv2.circle(output, (center_x, center_y), 5, (255, 0, 255), -1)
        label = f"CONE {position} {detection.confidence:.0%}"
        cv2.putText(
            output,
            label,
            (detection.x, max(24, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return output


def draw_panel_title(panel: np.ndarray, title: str) -> None:
    """Draw a readable title over either a camera or mask panel."""
    cv2.putText(panel, title, (17, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(panel, title, (17, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def make_dashboard(
    frame: np.ndarray,
    mask: np.ndarray,
    detections: list[Detection],
    navigator: SlalomNavigator,
    display_width: int = 1600,
) -> np.ndarray:
    """Put the annotated camera and red mask side by side under navigation text."""
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
    header = np.full((96, panels.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    command = f"ROBOT COMMAND: MOVE {navigator.direction}"
    command_color = (0, 220, 255) if navigator.direction == "RIGHT" else (255, 220, 0)
    text_size = cv2.getTextSize(command, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 3)[0]
    command_x = max(16, (header.shape[1] - text_size[0]) // 2)
    cv2.putText(
        header,
        command,
        (command_x, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        command_color,
        3,
    )

    detail = (
        f"{navigator.tracking_text(detections)}   |   "
        f"PASSED: {navigator.cones_passed}   |   NEXT: MOVE {navigator.next_direction}"
    )
    detail_size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
    detail_x = max(16, (header.shape[1] - detail_size[0]) // 2)
    cv2.putText(
        header,
        detail,
        (detail_x, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (225, 225, 225),
        2,
    )
    return np.vstack((header, panels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect red traffic cones with a webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--min-area", type=int, default=450, help="ignore smaller red regions")
    # Accepted for compatibility with the previous version. The mask is now
    # always shown beside the live camera.
    parser.add_argument("--show-mask", action="store_true", help=argparse.SUPPRESS)
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

    navigator = SlalomNavigator()
    print("Cone detector running. Press R to reset; Q or Escape to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                break

            detections = detect_cones(frame, min_area=args.min_area)
            mask = make_red_mask(frame)
            navigator.update(detections, frame.shape)
            dashboard = make_dashboard(
                frame, mask, detections, navigator, display_width=args.display_width
            )
            cv2.imshow("Red Cone Slalom Navigator", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                navigator = SlalomNavigator()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
