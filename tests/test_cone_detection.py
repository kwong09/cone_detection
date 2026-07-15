import unittest
import threading
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
    MOTOR_STOP_US,
    MotionPulseGate,
    apply_drive_output,
    choose_drive_command,
    course_complete_feedback,
    course_is_complete,
    draw_autonomous_controls,
    enforce_next_cone_timeout,
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


def bare_drive() -> FourEscDrive:
    """Build the motor state without importing Raspberry Pi hardware modules."""
    drive = FourEscDrive.__new__(FourEscDrive)
    drive._io_lock = threading.RLock()
    drive._motion_stop_timer = None
    drive._motion_deadline = None
    drive._watchdog_error = None
    drive._closed = False
    drive._current = list(MOTOR_STOP_US)
    drive._target = list(MOTOR_STOP_US)
    return drive


class ConeDetectionTests(unittest.TestCase):
    def test_creep_gate_starts_stopped_then_moves_at_forty_percent_duty(self) -> None:
        gate = MotionPulseGate(move_seconds=0.20, pause_seconds=0.30)
        planned = DriveCommand("FORWARD", (0.0001, 0.0001, 0.0001, 0.0001))

        first = gate.limit(planned, 0.0)
        just_before_move = gate.limit(planned, 0.299)
        move = gate.limit(planned, 0.30)
        move_deadline = gate.move_deadline
        just_before_pause = gate.limit(planned, 0.499)
        next_pause = gate.limit(planned, 0.50)

        self.assertAlmostEqual(gate.duty_cycle, 0.4)
        self.assertEqual(first.throttles, (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(first.name.startswith("VIEW PAUSE"))
        self.assertEqual(just_before_move.throttles, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(move, planned)
        self.assertAlmostEqual(move_deadline or 0.0, 0.50)
        self.assertEqual(just_before_pause, planned)
        self.assertEqual(next_pause.throttles, (0.0, 0.0, 0.0, 0.0))

    def test_creep_gate_uses_new_cone_plan_without_extending_pause(self) -> None:
        gate = MotionPulseGate(move_seconds=0.20, pause_seconds=0.20)
        right = DriveCommand("RIGHT", (0.003, 0.003, 0.0, 0.0))
        left = DriveCommand("LEFT", (0.0, 0.0, 0.003, 0.003))

        self.assertIn("NEXT RIGHT", gate.limit(right, 5.0).name)
        changed_during_pause = gate.limit(left, 5.10)
        self.assertEqual(changed_during_pause.name, "VIEW PAUSE - NEXT LEFT")
        self.assertEqual(gate.limit(left, 5.20), left)

    def test_safety_stop_resets_creep_gate_before_any_new_motion(self) -> None:
        gate = MotionPulseGate(move_seconds=0.20, pause_seconds=0.20)
        planned = DriveCommand("RIGHT", (0.003, 0.003, 0.0, 0.0))
        stopped = DriveCommand("STOP - CLOSE CONE", (0.0, 0.0, 0.0, 0.0))

        gate.limit(planned, 1.0)
        self.assertEqual(gate.limit(planned, 1.201), planned)
        self.assertEqual(gate.limit(stopped, 1.21), stopped)
        # Even though the old timing cycle would be in MOVE, a new positive
        # command must start with another camera-only pause.
        restarted = gate.limit(planned, 1.22)
        self.assertEqual(restarted.throttles, (0.0, 0.0, 0.0, 0.0))

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

    def test_zero_safety_command_bypasses_ramp_and_stops_immediately(self) -> None:
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
            DriveCommand("STOP - CLOSE CONE", (0.0, 0.0, 0.0, 0.0)),
            Namespace(turn_test_mode=False),
            course_complete=False,
        )

        self.assertEqual(events, [("stop", True)])

    def test_turn_command_stops_inside_pair_immediately(self) -> None:
        events: list[object] = []

        class FakeDrive:
            def stop(self, immediate: bool = False) -> None:
                events.append(("stop", immediate))

            def command(
                self,
                throttles: object,
                immediate_zero: bool = False,
                immediate_positive: bool = False,
                stop_deadline: float | None = None,
            ) -> None:
                events.append(
                    (
                        "command",
                        throttles,
                        immediate_zero,
                        immediate_positive,
                        stop_deadline,
                    )
                )

            def step(self) -> None:
                events.append("step")

        command = DriveCommand("RIGHT", (0.003, 0.003, 0.0, 0.0))
        apply_drive_output(
            FakeDrive(),  # type: ignore[arg-type]
            command,
            Namespace(turn_test_mode=False, creep_pause_seconds=0.20),
            course_complete=False,
        )

        self.assertEqual(
            events,
            [("command", command.throttles, True, True, None), "step"],
        )

    def test_creep_move_jumps_directly_to_minimum_motor_pulses(self) -> None:
        drive = bare_drive()

        drive.command(
            (0.003, 0.003, 0.0, 0.0),
            immediate_zero=True,
            immediate_positive=True,
        )

        self.assertEqual(drive._current, [1462, 1462, 1400, 1400])
        self.assertEqual(drive._target, [1462, 1462, 1400, 1400])

    def test_independent_deadline_stops_motors_without_another_camera_frame(self) -> None:
        drive = bare_drive()
        drive._current = [1462, 1462, 1400, 1400]
        drive._target = [1462, 1462, 1400, 1400]
        drive._motion_deadline = 7.0
        writes: list[tuple[int, int, int, int]] = []
        drive._write = lambda: writes.append(tuple(drive._current))  # type: ignore[method-assign]

        with patch("autonomous_cone_slalom.time.monotonic", return_value=7.01):
            drive._motion_deadline_expired(7.0)

        self.assertEqual(drive._current, list(MOTOR_STOP_US))
        self.assertEqual(drive._target, list(MOTOR_STOP_US))
        self.assertEqual(writes, [MOTOR_STOP_US])
        self.assertIsNone(drive._motion_deadline)

    def test_timed_stop_retries_one_failed_hardware_write(self) -> None:
        drive = bare_drive()
        drive._current = [1462, 1462, 1400, 1400]
        drive._target = [1462, 1462, 1400, 1400]
        drive._motion_deadline = 8.0
        attempts = 0

        def flaky_write() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("temporary I2C failure")

        drive._write = flaky_write  # type: ignore[method-assign]
        with patch("autonomous_cone_slalom.time.monotonic", return_value=8.01):
            drive._motion_deadline_expired(8.0)

        self.assertEqual(attempts, 2)
        self.assertIsNone(drive._watchdog_error)
        self.assertEqual(drive._current, list(MOTOR_STOP_US))

    def test_absolute_deadlines_prevent_slow_loop_from_extending_move(self) -> None:
        gate = MotionPulseGate(move_seconds=0.20, pause_seconds=0.20)
        planned = DriveCommand("FORWARD", (0.0001, 0.0001, 0.0001, 0.0001))
        gate.limit(planned, 0.0)

        for now, expected_deadline in (
            (0.201, 0.4),
            (0.601, 0.8),
            (1.001, 1.2),
        ):
            with self.subTest(now=now):
                self.assertEqual(gate.limit(planned, now), planned)
                self.assertAlmostEqual(gate.move_deadline or 0.0, expected_deadline)

    def test_visual_height_does_not_turn_before_ideal_distance(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        centered_close = Detection(540, 310, 200, 230, 0.95)
        navigator = CameraMotionSlalomNavigator(
            turn_start_cm=130.0,
            turn_start_height_ratio=0.30,
        )

        first_feedback = navigator.update(
            [centered_close], calibration, FRAME_SHAPE
        )
        self.assertEqual(navigator.phase, "APPROACHING")
        self.assertIn("FORWARD", first_feedback)

        second_feedback = navigator.update([centered_close], calibration, FRAME_SHAPE)
        self.assertEqual(navigator.phase, "APPROACHING")
        self.assertIsNone(navigator.turn_trigger_source)
        self.assertIn("FORWARD", second_feedback)
        self.assertFalse(navigator.pass_armed)
        self.assertEqual(navigator.cones_passed, 0)

        command = choose_drive_command(
            navigator,
            FRAME_SHAPE[1],
            Namespace(
                max_cones=3,
                cruise_throttle=0.0001,
                turn_outside_throttle=0.0015,
                turn_inside_throttle=0.0,
                turn_test_mode=False,
            ),
        )
        self.assertEqual(command.name, "FORWARD")
        self.assertEqual(command.throttles, (0.0001,) * 4)

    def test_turn_starts_only_at_ideal_distance(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        far = Detection(540, 310, 200, 230, 0.95)
        ideal_distance = Detection(440, 200, 400, 500, 0.95)
        navigator = CameraMotionSlalomNavigator(turn_start_cm=130.0)

        navigator.update([far], calibration, FRAME_SHAPE)
        self.assertEqual(navigator.phase, "APPROACHING")
        feedback = navigator.update([ideal_distance], calibration, FRAME_SHAPE)
        self.assertEqual(navigator.phase, "TURNING")
        self.assertEqual(navigator.turn_trigger_source, "DISTANCE")
        self.assertIn("BEGIN SLOW RIGHT TURN", feedback)

    def test_visual_size_never_bypasses_distance_at_any_resolution(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )

        for frame_shape in ((720, 1280, 3), (1080, 1920, 3)):
            with self.subTest(frame_height=frame_shape[0]):
                target_height = round(frame_shape[0] * 0.31)
                target_width = round(frame_shape[1] * 0.12)
                target = Detection(
                    (frame_shape[1] - target_width) // 2,
                    round(frame_shape[0] * 0.42),
                    target_width,
                    target_height,
                    0.95,
                )
                navigator = CameraMotionSlalomNavigator(
                    turn_start_height_ratio=0.30
                )

                navigator.update([target], calibration, frame_shape)
                navigator.update([target], calibration, frame_shape)

                self.assertEqual(navigator.phase, "APPROACHING")
                self.assertIsNone(navigator.turn_trigger_source)

    def test_left_mounted_camera_corrects_to_vehicle_centerline(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=1000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        # At about 130 cm, a cone on the vehicle centerline appears roughly
        # 59 pixels right of image center because the camera is 7.62 cm left.
        vehicle_centerline_cone = Detection(659, 300, 80, 235, 0.95)
        navigator = CameraMotionSlalomNavigator(camera_offset_cm=-7.62)

        navigator.update([vehicle_centerline_cone], calibration, FRAME_SHAPE)

        self.assertAlmostEqual(
            navigator.corrected_center_x_ratio or 0.0,
            0.5,
            delta=0.002,
        )

    def test_next_cone_selection_uses_vehicle_center_not_camera_center(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=1000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        image_centered = Detection(600, 300, 80, 100, 0.95)
        # At this cone's roughly 305 cm range, vehicle center is about 25 px
        # right of the left-mounted camera's image center.
        vehicle_centered = Detection(625, 300, 80, 100, 0.95)
        navigator = CameraMotionSlalomNavigator(
            camera_offset_cm=-7.62,
            awaiting_new_cone=True,
            phase="SEARCHING",
        )

        navigator.update(
            [image_centered, vehicle_centered],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertIs(navigator.current_target, vehicle_centered)

    def test_cropped_cone_immediately_triggers_turn_despite_distance_smoothing(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        cropped = Detection(500, 360, 280, 360, 0.98, cropped=True)
        navigator = CameraMotionSlalomNavigator(
            phase="APPROACHING",
            smoothed_distance_cm=300.0,
        )

        feedback = navigator.update([cropped], calibration, FRAME_SHAPE)

        self.assertEqual(navigator.raw_distance_cm, navigator.hard_turn_cm)
        self.assertGreater(navigator.smoothed_distance_cm or 0.0, navigator.turn_start_cm)
        self.assertEqual(navigator.phase, "TURNING")
        self.assertEqual(navigator.turn_trigger_source, "DISTANCE")
        self.assertIn("BEGIN SLOW RIGHT TURN", feedback)
        self.assertNotIn("HARD", feedback)
        self.assertIn("TURN", feedback)
        self.assertNotIn("FORWARD", feedback)

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

    def test_countersteer_counts_only_frames_with_motor_output(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            phase="COUNTERSTEERING",
            countersteer_frames=2,
            countersteer_remaining=2,
            awaiting_new_cone=True,
        )

        navigator.update([], calibration, FRAME_SHAPE, motion_applied=False)
        self.assertEqual(navigator.countersteer_remaining, 2)
        navigator.update([], calibration, FRAME_SHAPE, motion_applied=True)
        self.assertEqual(navigator.countersteer_remaining, 1)
        navigator.update([], calibration, FRAME_SHAPE, motion_applied=False)
        self.assertEqual(navigator.countersteer_remaining, 1)
        navigator.update([], calibration, FRAME_SHAPE, motion_applied=True)
        self.assertEqual(navigator.countersteer_remaining, 0)
        self.assertEqual(navigator.phase, "SEARCHING")

    def test_countersteer_accepts_visible_next_cone_on_final_frame(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            phase="COUNTERSTEERING",
            countersteer_remaining=1,
            awaiting_new_cone=True,
        )
        distant_left_cone = Detection(100, 300, 80, 100, 0.95)

        navigator.update(
            [distant_left_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=True,
        )

        self.assertEqual(navigator.countersteer_remaining, 0)
        self.assertIs(navigator.current_target, distant_left_cone)
        self.assertFalse(navigator.awaiting_new_cone)
        self.assertEqual(navigator.phase, "APPROACHING")

    def test_countersteer_view_pause_accepts_safe_forward_next_cone(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            phase="COUNTERSTEERING",
            countersteer_remaining=12,
            awaiting_new_cone=True,
        )
        distant_forward_cone = Detection(600, 300, 80, 100, 0.95)

        navigator.update(
            [distant_forward_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertIs(navigator.current_target, distant_forward_cone)
        self.assertEqual(navigator.countersteer_remaining, 0)
        self.assertFalse(navigator.awaiting_new_cone)

    def test_countersteer_does_not_reacquire_far_old_edge_cone_early(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(direction_index=0)
        navigator._finish_pass()
        # The previous RIGHT pass leaves the old cone on the left. At x=35%
        # it is still fairly central, so a simple edge crop would not reject it.
        distant_old_edge_cone = Detection(398, 300, 100, 100, 0.95)

        navigator.update(
            [distant_old_edge_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertIsNone(navigator.current_target)
        self.assertEqual(navigator.countersteer_remaining, 12)
        self.assertTrue(navigator.awaiting_new_cone)

    def test_countersteer_ignores_close_cone_on_side_just_passed(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        cases = (
            # Previous RIGHT pass: old cone remains at x=35% on the left.
            (0, Detection(348, 310, 200, 230, 0.98), "LEFT"),
            # Previous LEFT pass: old cone remains at x=65% on the right.
            (1, Detection(732, 310, 200, 230, 0.98), "RIGHT"),
        )

        for starting_direction, old_cone, ignored_side in cases:
            with self.subTest(ignored_side=ignored_side):
                navigator = CameraMotionSlalomNavigator(
                    direction_index=starting_direction
                )
                navigator._finish_pass()
                feedback = navigator.update(
                    [old_cone],
                    calibration,
                    FRAME_SHAPE,
                    motion_applied=False,
                )

                self.assertEqual(
                    navigator.passed_cone_side_to_ignore,
                    ignored_side,
                )
                self.assertFalse(navigator.close_cone_hazard)
                self.assertEqual(navigator.countersteer_remaining, 12)
                self.assertIn("CONE 1 PASSED", feedback)

    def test_new_target_does_not_switch_back_to_larger_passed_cone(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(direction_index=0)
        navigator._finish_pass()
        next_cone = Detection(600, 300, 80, 100, 0.95)
        old_larger_left_cone = Detection(248, 300, 300, 300, 0.98, cropped=True)

        navigator.update(
            [next_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )
        self.assertIs(navigator.current_target, next_cone)

        next_cone_again = Detection(602, 301, 82, 102, 0.95)
        navigator.update(
            [old_larger_left_cone, next_cone_again],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertIs(navigator.current_target, next_cone_again)
        self.assertFalse(navigator.close_cone_hazard)

    def test_accepted_next_cone_tracks_across_previous_cone_side(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(direction_index=0)
        navigator._finish_pass()
        centered_next_cone = Detection(600, 300, 80, 100, 0.95)
        navigator.update(
            [centered_next_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )
        self.assertFalse(navigator.awaiting_new_cone)

        # Camera yaw can carry the accepted target into the left half that was
        # excluded only while cone 1 was still being filtered out.
        shifted_same_cone = Detection(434, 301, 80, 100, 0.95)
        navigator.update(
            [shifted_same_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertIs(navigator.current_target, shifted_same_cone)
        self.assertEqual(navigator.phase, "APPROACHING")

    def test_countersteer_stops_for_ambiguous_close_centered_cone(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            phase="COUNTERSTEERING",
            countersteer_remaining=12,
            awaiting_new_cone=True,
        )
        close_centered = Detection(540, 310, 200, 230, 0.98)

        feedback = navigator.update(
            [close_centered],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )

        self.assertTrue(navigator.close_cone_hazard)
        self.assertEqual(navigator.countersteer_remaining, 0)
        self.assertTrue(feedback.startswith("STOP"))

    def test_next_cone_found_while_stopped_controls_next_move(self) -> None:
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            phase="SEARCHING",
            awaiting_new_cone=True,
        )
        command_args = Namespace(
            max_cones=3,
            cruise_throttle=0.0001,
            turn_outside_throttle=0.0015,
            turn_inside_throttle=0.0,
            turn_test_mode=False,
        )
        gate = MotionPulseGate(move_seconds=0.20, pause_seconds=0.20)

        search_right = choose_drive_command(navigator, FRAME_SHAPE[1], command_args)
        self.assertEqual(search_right.name, "RIGHT")
        self.assertEqual(
            gate.limit(search_right, 3.0).throttles,
            (0.0, 0.0, 0.0, 0.0),
        )

        distant_left_cone = Detection(100, 300, 80, 100, 0.95)
        navigator.update(
            [distant_left_cone],
            calibration,
            FRAME_SHAPE,
            motion_applied=False,
        )
        approach_next = choose_drive_command(navigator, FRAME_SHAPE[1], command_args)
        self.assertEqual(approach_next.name, "FORWARD")
        self.assertEqual(
            gate.limit(approach_next, 3.10).name,
            "VIEW PAUSE - NEXT FORWARD",
        )
        self.assertEqual(gate.limit(approach_next, 3.20), approach_next)

    def test_next_cone_timeout_includes_countersteering_time(self) -> None:
        command = DriveCommand("LEFT", (0.0, 0.0, 0.003, 0.003))
        navigator = Namespace(
            awaiting_new_cone=True,
            close_cone_hazard=False,
            countersteer_remaining=12,
        )

        limited, started_at = enforce_next_cone_timeout(
            command,
            navigator,
            paused=False,
            started_at=None,
            now=10.0,
            timeout_seconds=4.0,
        )
        self.assertEqual(limited, command)
        timed_out, started_at = enforce_next_cone_timeout(
            command,
            navigator,
            paused=False,
            started_at=started_at,
            now=14.0,
            timeout_seconds=4.0,
        )
        self.assertEqual(timed_out.name, "STOP - NEXT CONE NOT FOUND")
        self.assertEqual(timed_out.throttles, (0.0, 0.0, 0.0, 0.0))

        navigator.awaiting_new_cone = False
        resumed, cleared_start = enforce_next_cone_timeout(
            command,
            navigator,
            paused=False,
            started_at=started_at,
            now=14.1,
            timeout_seconds=4.0,
        )
        self.assertEqual(resumed, command)
        self.assertIsNone(cleared_start)

    def test_three_cone_course_uses_ultra_slow_defaults(self) -> None:
        with patch("sys.argv", ["combined_cone_detection_slalom.py"]):
            args = parse_autonomous_args()

        self.assertEqual(args.max_cones, 3)
        self.assertEqual(args.turn_start_height_ratio, 0.30)
        self.assertEqual(args.countersteer_frames, 12)
        self.assertEqual(args.cruise_throttle, 0.003)
        self.assertEqual(args.turn_outside_throttle, 0.0045)
        self.assertEqual(args.turn_inside_throttle, 0.0)
        self.assertEqual(args.robot_width_cm, 30.48)
        self.assertEqual(args.camera_from_left_cm, 7.62)
        self.assertEqual(args.ramp_step_us, 3)
        self.assertEqual(args.creep_move_seconds, 0.20)
        self.assertEqual(args.creep_pause_seconds, 0.30)
        self.assertEqual(args.search_timeout_seconds, 4.0)
        self.assertEqual(
            FourEscDrive._pulse_for_throttle(0, args.cruise_throttle),
            1462,
        )
        self.assertEqual(
            FourEscDrive._pulse_for_throttle(0, args.turn_outside_throttle),
            1463,
        )
        self.assertEqual(
            FourEscDrive._pulse_for_throttle(0, args.turn_inside_throttle),
            1400,
        )
        navigator = new_navigator(args)
        self.assertEqual(navigator.max_cones, 3)
        self.assertAlmostEqual(navigator.camera_offset_cm, -7.62)
        self.assertIsNone(new_preview_navigator(args).max_cones)

    def test_turn_test_mode_makes_left_and_right_motor_groups_distinct(self) -> None:
        command_args = Namespace(
            cruise_throttle=0.08,
            turn_outside_throttle=0.10,
            turn_inside_throttle=0.04,
            turn_test_mode=True,
        )

        def navigator_for(direction: str, phase: str = "TURNING") -> Namespace:
            return Namespace(
                current_target=Detection(540, 200, 200, 300, 0.9),
                phase=phase,
                close_cone_hazard=False,
                countersteer_remaining=0,
                awaiting_new_cone=False,
                direction=direction,
                smoothed_distance_cm=180.0,
                clearance_seen=False,
            )

        left = choose_drive_command(
            navigator_for("LEFT"),
            FRAME_SHAPE[1],
            command_args,
        )
        right = choose_drive_command(
            navigator_for("RIGHT"),
            FRAME_SHAPE[1],
            command_args,
        )

        self.assertEqual(left.name, "LEFT")
        self.assertEqual(left.throttles, (0.0, 0.0, 0.10, 0.10))
        self.assertEqual(right.name, "RIGHT")
        self.assertEqual(right.throttles, (0.10, 0.10, 0.0, 0.0))

        approach = choose_drive_command(
            navigator_for("RIGHT", phase="APPROACHING"),
            FRAME_SHAPE[1],
            command_args,
        )
        self.assertEqual(approach.name, "APPROACH - NO TURN")
        self.assertEqual(approach.throttles, (0.0, 0.0, 0.0, 0.0))

        command_args.turn_test_mode = False
        normal_approach = choose_drive_command(
            navigator_for("LEFT", phase="APPROACHING"),
            FRAME_SHAPE[1],
            command_args,
        )
        self.assertEqual(normal_approach.name, "FORWARD")
        self.assertEqual(normal_approach.throttles, (0.08, 0.08, 0.08, 0.08))

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

    def test_turn_test_banner_distinguishes_automatic_camera_pause(self) -> None:
        title, detail, _ = turn_test_banner(
            DriveCommand(
                "VIEW PAUSE - NEXT RIGHT",
                (0.0, 0.0, 0.0, 0.0),
            ),
            paused=False,
        )
        self.assertIn("CAMERA VIEW PAUSE", title)
        self.assertIn("ALL STOP", title)
        self.assertIn("NEXT MOTION COMMAND: RIGHT", detail)

    def test_camera_pause_footer_fits_1200_pixel_preview(self) -> None:
        dashboard = np.zeros((720, 1200, 3), dtype=np.uint8)
        text_calls: list[tuple[str, tuple[int, int], float, int]] = []

        def record_text(*args: object, **_kwargs: object) -> None:
            text_calls.append((args[1], args[2], args[4], args[6]))  # type: ignore[arg-type]

        with patch("autonomous_cone_slalom.cv2.putText", side_effect=record_text):
            draw_autonomous_controls(
                dashboard,
                paused=False,
                command=DriveCommand(
                    "VIEW PAUSE - NEXT RIGHT",
                    (0.0, 0.0, 0.0, 0.0),
                ),
                turn_test_mode=False,
                cones_passed=2,
                max_cones=3,
                course_complete=False,
                creep_duty_cycle=0.5,
            )

        self.assertEqual(len(text_calls), 1)
        text, origin, scale, thickness = text_calls[0]
        rendered_width = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )[0][0]
        self.assertLessEqual(origin[0] + rendered_width, dashboard.shape[1])

    def test_safety_stop_footer_never_claims_auto_move(self) -> None:
        dashboard = np.zeros((720, 1200, 3), dtype=np.uint8)
        labels: list[str] = []

        def record_text(*args: object, **_kwargs: object) -> None:
            labels.append(args[1])  # type: ignore[arg-type]

        with patch("autonomous_cone_slalom.cv2.putText", side_effect=record_text):
            draw_autonomous_controls(
                dashboard,
                paused=False,
                command=DriveCommand(
                    "STOP - CLOSE CONE",
                    (0.0, 0.0, 0.0, 0.0),
                ),
                turn_test_mode=False,
                cones_passed=1,
                max_cones=3,
                course_complete=False,
                creep_duty_cycle=0.5,
            )

        self.assertEqual(len(labels), 1)
        self.assertIn("AUTO STOP", labels[0])
        self.assertIn("ALL MOTORS STOPPED", labels[0])
        self.assertNotIn("AUTO MOVE", labels[0])

    def test_turn_test_zero_channels_jump_to_stop_pulse(self) -> None:
        drive = bare_drive()
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

    def test_awaiting_navigator_rejects_visually_close_cone_with_bad_range(self) -> None:
        visible = Detection(540, 310, 200, 230, 0.98)
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            awaiting_new_cone=True,
            phase="SEARCHING",
            turn_start_height_ratio=0.30,
        )

        feedback = navigator.update([visible], calibration, FRAME_SHAPE)

        self.assertIsNone(navigator.current_target)
        self.assertTrue(navigator.awaiting_new_cone)
        self.assertTrue(navigator.close_cone_hazard)
        self.assertTrue(feedback.startswith("STOP"))

    def test_awaiting_navigator_ignores_close_edge_cone_with_bad_range(self) -> None:
        passed_edge_cone = Detection(20, 310, 200, 230, 0.98)
        calibration = CameraCalibration(
            focal_length_px=2000.0,
            cone_height_cm=30.5,
            frame_width=1280,
            frame_height=720,
            calibration_distance_cm=100.0,
        )
        navigator = CameraMotionSlalomNavigator(
            awaiting_new_cone=True,
            phase="SEARCHING",
            turn_start_height_ratio=0.30,
        )

        navigator.update([passed_edge_cone], calibration, FRAME_SHAPE)

        self.assertIsNone(navigator.current_target)
        self.assertTrue(navigator.awaiting_new_cone)
        self.assertFalse(navigator.close_cone_hazard)

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
