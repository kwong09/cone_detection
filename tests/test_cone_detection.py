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
    new_navigator,
)
from autonomous_cone_slalom import (
    DriveCommand,
    FourEscDrive,
    apply_drive_output,
    choose_drive_command,
    course_complete_feedback,
    course_is_complete,
    main as autonomous_main,
    new_preview_navigator,
    parse_args as parse_autonomous_args,
    synchronize_preview_course,
    turn_test_banner,
)


FRAME_SHAPE = (720, 1280, 3)


def polygon_frame(points: list[tuple[int, int]], color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], color)
    return frame


class ConeDetectionTests(unittest.TestCase):
    def test_paused_preview_cannot_change_displayed_course_progress(self) -> None:
        preview = Namespace(cones_passed=2, direction_index=1)
        navigator = Namespace(cones_passed=1, direction_index=0)

        synchronize_preview_course(preview, navigator)

        self.assertEqual(preview.cones_passed, 1)
        self.assertEqual(preview.direction_index, 0)

    def test_course_completion_bypasses_ramp_and_stops_immediately(self) -> None:
        events: list[object] = []

        class FakeDrive:
            def stop(self, immediate: bool = False) -> None:
                events.append(("stop", immediate))

            def command(self, *_args: object, **_kwargs: object) -> None:
                events.append("command")

            def step(self) -> None:
                events.append("step")

        apply_drive_output(
            FakeDrive(),  # type: ignore[arg-type]
            DriveCommand("COURSE COMPLETE", (0.0, 0.0, 0.0, 0.0)),
            Namespace(turn_test_mode=False),
            course_complete=True,
        )

        self.assertEqual(events, [("stop", True)])

    def test_third_pass_latches_complete_without_countersteering(self) -> None:
        navigator = CameraMotionSlalomNavigator(
            max_cones=3,
            cones_passed=2,
            direction_index=0,
            countersteer_frames=12,
        )

        navigator._finish_pass()

        self.assertEqual(navigator.cones_passed, 3)
        self.assertTrue(navigator.course_complete)
        self.assertEqual(navigator.phase, "COMPLETE")
        self.assertEqual(navigator.countersteer_remaining, 0)
        self.assertFalse(navigator.awaiting_new_cone)
        self.assertEqual(navigator.direction_index, 0)
        self.assertIn("COURSE COMPLETE", navigator.update([], None, FRAME_SHAPE))

        args = Namespace(max_cones=3)
        self.assertTrue(course_is_complete(navigator, args))
        self.assertIn("3 CONES PASSED", course_complete_feedback(args))
        command = choose_drive_command(navigator, FRAME_SHAPE[1], args)
        self.assertEqual(command.throttles, (0.0, 0.0, 0.0, 0.0))

        title, _, _ = turn_test_banner(
            DriveCommand("COURSE COMPLETE", (0.0, 0.0, 0.0, 0.0)),
            paused=True,
        )
        self.assertIn("COURSE COMPLETE", title)

    def test_second_pass_still_alternates_and_countersteers(self) -> None:
        navigator = CameraMotionSlalomNavigator(
            max_cones=3,
            cones_passed=1,
            direction_index=0,
            countersteer_frames=12,
        )

        navigator._finish_pass()

        self.assertEqual(navigator.cones_passed, 2)
        self.assertFalse(navigator.course_complete)
        self.assertEqual(navigator.direction_index, 1)
        self.assertEqual(navigator.countersteer_remaining, 12)
        self.assertTrue(navigator.awaiting_new_cone)

    def test_three_cone_course_uses_slower_defaults(self) -> None:
        with patch("sys.argv", ["combined_cone_detection_slalom.py"]):
            args = parse_autonomous_args()

        self.assertEqual(args.max_cones, 3)
        self.assertEqual(args.cruise_throttle, 0.10)
        self.assertEqual(args.turn_outside_throttle, 0.12)
        self.assertEqual(args.turn_inside_throttle, 0.04)
        self.assertEqual(args.ramp_step_us, 10)
        self.assertEqual(new_navigator(args).max_cones, 3)
        self.assertIsNone(new_preview_navigator(args).max_cones)

    def test_turn_test_mode_makes_left_and_right_motor_groups_distinct(self) -> None:
        command_args = Namespace(
            cruise_throttle=0.08,
            turn_outside_throttle=0.10,
            turn_inside_throttle=0.04,
            hard_inside_throttle=0.0,
            hard_turn_cm=80.0,
            turn_test_mode=True,
        )

        def navigator_for(target: Detection) -> Namespace:
            return Namespace(
                current_target=target,
                phase="APPROACHING",
                close_cone_hazard=False,
                countersteer_remaining=0,
                awaiting_new_cone=False,
                direction="RIGHT",
                smoothed_distance_cm=180.0,
                clearance_seen=False,
            )

        left = choose_drive_command(
            navigator_for(Detection(100, 200, 200, 300, 0.9)),
            FRAME_SHAPE[1],
            command_args,
        )
        right = choose_drive_command(
            navigator_for(Detection(980, 200, 200, 300, 0.9)),
            FRAME_SHAPE[1],
            command_args,
        )

        self.assertEqual(left.name, "LEFT")
        self.assertEqual(left.throttles, (0.0, 0.0, 0.10, 0.10))
        self.assertEqual(right.name, "RIGHT")
        self.assertEqual(right.throttles, (0.10, 0.10, 0.0, 0.0))

        centered = choose_drive_command(
            navigator_for(Detection(540, 200, 200, 300, 0.9)),
            FRAME_SHAPE[1],
            command_args,
        )
        self.assertEqual(centered.name, "CENTERED - NO TURN")
        self.assertEqual(centered.throttles, (0.0, 0.0, 0.0, 0.0))

        close_turn = navigator_for(Detection(540, 200, 200, 300, 0.9))
        close_turn.phase = "TURNING"
        close_turn.smoothed_distance_cm = 100.0
        planned_right = choose_drive_command(
            close_turn,
            FRAME_SHAPE[1],
            command_args,
        )
        self.assertEqual(planned_right.name, "RIGHT")
        self.assertEqual(planned_right.throttles, (0.10, 0.10, 0.0, 0.0))

        command_args.turn_test_mode = False
        normal_left = choose_drive_command(
            navigator_for(Detection(100, 200, 200, 300, 0.9)),
            FRAME_SHAPE[1],
            command_args,
        )
        self.assertEqual(normal_left.throttles, (0.04, 0.04, 0.10, 0.10))

        left_title, left_detail, _ = turn_test_banner(left, paused=False)
        right_title, right_detail, _ = turn_test_banner(right, paused=False)
        self.assertIn("LEFT", left_title)
        self.assertIn("MOTORS 3-4 RUN", left_detail)
        self.assertIn("RIGHT", right_title)
        self.assertIn("MOTORS 1-2 RUN", right_detail)

    def test_turn_test_banner_shows_all_motors_stopped_while_paused(self) -> None:
        title, detail, _ = turn_test_banner(
            DriveCommand("PAUSED", (0.0, 0.0, 0.0, 0.0)),
            paused=True,
        )
        self.assertIn("PAUSED", title)
        self.assertIn("LEFT = M3-4", detail)
        self.assertIn("FAR CENTER", detail)

    def test_turn_test_zero_channels_jump_to_stop_pulse(self) -> None:
        drive = FourEscDrive.__new__(FourEscDrive)
        drive._current = [1524, 1524, 1400, 1400]
        drive._target = [1524, 1524, 1400, 1400]

        drive.command((0.0, 0.0, 0.10, 0.10), immediate_zero=True)

        self.assertEqual(drive._current, [1400, 1400, 1400, 1400])
        self.assertEqual(drive._target, [1400, 1400, 1524, 1524])

    def test_turn_test_mode_rejects_headless_immediate_start(self) -> None:
        with patch(
            "sys.argv",
            ["combined_cone_detection_slalom.py", "--drive", "--turn-test-mode", "--headless"],
        ):
            with self.assertRaisesRegex(SystemExit, "requires the live dashboard"):
                autonomous_main()

    def test_drive_stops_before_calibration_and_closes_on_setup_failure(self) -> None:
        events: list[str] = []

        class FakeDrive:
            def arm(self, _seconds: float) -> None:
                events.append("arm")

            def close(self) -> None:
                events.append("close")

        def make_drive(_ramp_step_us: int) -> FakeDrive:
            events.append("drive-stop")
            return FakeDrive()

        def missing_calibration(_path: Path) -> None:
            events.append("calibration")
            return None

        with (
            patch("sys.argv", ["combined_cone_detection_slalom.py", "--drive"]),
            patch("autonomous_cone_slalom.FourEscDrive", side_effect=make_drive),
            patch(
                "autonomous_cone_slalom.CameraCalibration.load",
                side_effect=missing_calibration,
            ),
            patch("autonomous_cone_slalom.atexit.register"),
            patch("autonomous_cone_slalom.signal.signal"),
        ):
            with self.assertRaisesRegex(SystemExit, "No Pi slalom calibration"):
                autonomous_main()

        self.assertEqual(events, ["drive-stop", "arm", "calibration", "close"])

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
