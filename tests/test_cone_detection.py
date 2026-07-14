import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from cone_detection_iteration_1 import Detection, detect_cones
from cone_detection_iteration_3 import CameraCalibration
from cone_detection_iteration_4 import CameraMotionSlalomNavigator
from cone_detection_iteration_5_pi import (
    calibrate_from_multiple_frames,
    detections_for_dashboard,
)
from autonomous_cone_slalom import choose_drive_command


FRAME_SHAPE = (720, 1280, 3)


def polygon_frame(points: list[tuple[int, int]], color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], color)
    return frame


class ConeDetectionTests(unittest.TestCase):
    def test_dashboard_keeps_visible_cone_when_navigator_has_no_target(self) -> None:
        visible = Detection(304, 443, 248, 277, 0.98, cropped=True)
        self.assertEqual(detections_for_dashboard([visible], None), [visible])

    def test_awaiting_navigator_stops_for_ambiguous_close_cone(self) -> None:
        visible = Detection(304, 443, 248, 277, 0.98, cropped=True)
        calibration = CameraCalibration(
            focal_length_px=4000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            turn_start_cm=130.0,
            hard_turn_cm=80.0,
            pass_distance_cm=60.0,
            awaiting_new_cone=True,
        )
        feedback = navigator.update([visible], calibration, FRAME_SHAPE)
        self.assertIsNone(navigator.current_target)
        self.assertTrue(navigator.awaiting_new_cone)
        self.assertTrue(navigator.close_cone_hazard)
        self.assertTrue(feedback.startswith("STOP"))

        command_args = Namespace(
            cruise_throttle=0.12,
            turn_outside_throttle=0.12,
            turn_inside_throttle=0.07,
            hard_inside_throttle=0.0,
            hard_turn_cm=80.0,
        )
        command = choose_drive_command(navigator, FRAME_SHAPE[1], command_args)
        self.assertEqual(command.name, "STOP - CLOSE CONE")
        self.assertEqual(command.throttles, (0.0, 0.0, 0.0, 0.0))

        # A single missed detection must not restart a blind search turn.
        feedback = navigator.update([], calibration, FRAME_SHAPE)
        self.assertTrue(navigator.close_cone_hazard)
        self.assertTrue(feedback.startswith("STOP"))
        command = choose_drive_command(navigator, FRAME_SHAPE[1], command_args)
        self.assertEqual(command.name, "STOP - CLOSE CONE")
        self.assertEqual(command.throttles, (0.0, 0.0, 0.0, 0.0))

    def test_detects_complete_distant_cone(self) -> None:
        frame = polygon_frame(
            [(640, 180), (500, 620), (780, 620)],
            (0, 0, 255),
        )
        self.assertEqual(len(detect_cones(frame)), 1)

    def test_detects_large_gently_tapered_close_cone(self) -> None:
        # This bottom-cropped shape deliberately fails the distant cone's
        # aspect and taper strictness, exercising the close-range path.
        frame = polygon_frame(
            [(440, 250), (840, 250), (880, 719), (400, 719)],
            (0, 0, 255),
        )
        detections = detect_cones(frame)
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(detections[0].y + detections[0].height, 719)
        self.assertTrue(detections[0].cropped)

    def test_detects_complete_close_cone_that_is_wider_than_tall(self) -> None:
        frame = polygon_frame(
            [(465, 230), (815, 230), (940, 680), (340, 680)],
            (0, 0, 255),
        )
        detections = detect_cones(frame)
        self.assertEqual(len(detections), 1)
        self.assertFalse(detections[0].cropped)

    def test_cropped_cone_forces_conservative_navigation_distance(self) -> None:
        detection = detect_cones(
            polygon_frame(
                [(440, 250), (840, 250), (880, 719), (400, 719)],
                (0, 0, 255),
            )
        )[0]
        calibration = CameraCalibration(
            focal_length_px=4000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            turn_start_cm=130.0,
            hard_turn_cm=80.0,
            pass_distance_cm=60.0,
        )
        navigator.update([detection], calibration, FRAME_SHAPE)
        self.assertEqual(navigator.raw_distance_cm, 80.0)
        self.assertEqual(navigator.phase, "TURNING")

    def test_detects_dim_red_after_close_exposure_change(self) -> None:
        hsv = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        cv2.fillPoly(
            hsv,
            [np.array([(640, 180), (500, 620), (780, 620)], dtype=np.int32)],
            (3, 130, 60),
        )
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(len(detect_cones(frame)), 1)

    def test_rejects_large_red_rectangle(self) -> None:
        frame = polygon_frame(
            [(450, 250), (830, 250), (830, 719), (450, 719)],
            (0, 0, 255),
        )
        self.assertEqual(detect_cones(frame), [])

    def test_rejects_large_red_circle(self) -> None:
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        cv2.circle(frame, (640, 500), 210, (0, 0, 255), -1)
        self.assertEqual(detect_cones(frame), [])

    def test_rejects_bottom_cropped_red_ball(self) -> None:
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        cv2.circle(frame, (640, 650), 210, (0, 0, 255), -1)
        self.assertEqual(detect_cones(frame), [])

    def test_rejects_known_skin_edge_hue(self) -> None:
        hsv = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        cv2.fillPoly(
            hsv,
            [np.array([(640, 180), (500, 620), (780, 620)], dtype=np.int32)],
            (4, 200, 180),
        )
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(detect_cones(frame), [])

    def test_headless_calibration_rejects_cropped_cone_frames(self) -> None:
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

        class FakeCamera:
            def read(self) -> tuple[bool, np.ndarray]:
                return True, frame

        args = Namespace(
            calibration_distance_cm=100.0,
            cone_height_cm=30.5,
            min_area=450,
            calibration_file=Path("unused-calibration.json"),
        )
        cropped = Detection(300, 300, 500, 420, 0.9, cropped=True)
        with patch(
            "cone_detection_iteration_5_pi.detect_cones",
            return_value=[cropped],
        ):
            with self.assertRaisesRegex(SystemExit, "fewer than 10"):
                calibrate_from_multiple_frames(FakeCamera(), args)


if __name__ == "__main__":
    unittest.main()
