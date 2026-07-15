#!/usr/bin/env python3
"""Drive the four-ESC robot through an alternating red-cone slalom.

This combines the team's PCA9685 motor calibration with the Raspberry Pi 5 /
Arducam detector and the camera-motion-aware navigator.  Motor output remains
forward-only: turns are made by slowing the motors on the inside of the turn.

Keep the wheels raised for the first test and keep a physical kill switch in
reach.  Camera loss, missing calibration, Ctrl+C, and normal exit all command
the four ESCs to their calibrated stop pulses.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from cone_detection_iteration_1 import detect_cones, make_red_mask
from cone_detection_iteration_3 import CameraCalibration, make_dashboard
from cone_detection_iteration_5_pi import (
    create_camera,
    detections_for_dashboard,
    new_navigator,
    validate_args as validate_camera_args,
)


I2C_ADDRESS = 0x40
PWM_FREQUENCY = 50
ESC_CHANNELS = (0, 1, 2, 3)

# These values come directly from the supplied four-ESC controller.
MOTOR_STOP_US = (1400, 1400, 1400, 1400)
MOTOR_START_US = (1460, 1460, 1460, 1460)

# The original driver clamps pulses to 2100 us even though its UI says 2500.
# Preserve that tested electrical limit until each ESC has been calibrated.
MOTOR_MAX_US = (2100, 2100, 2100, 2100)

# In the supplied controller, Right activates 1-2 and Left activates 3-4.
# These names describe the requested turn, not assumed chassis wiring.
RIGHT_TURN_MOTORS = (0, 1)
LEFT_TURN_MOTORS = (2, 3)
WINDOW_NAME = "Autonomous Cone Slalom - Live Camera"


class FourEscDrive:
    """Forward-only drive using the robot's measured ESC pulse values."""

    def __init__(self, ramp_step_us: int) -> None:
        self._pca = None
        self._closed = True
        self._io_lock = threading.RLock()
        self._motion_stop_timer: threading.Timer | None = None
        self._motion_deadline: float | None = None
        self._watchdog_error: Exception | None = None
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError as error:
            raise RuntimeError(
                "Missing PCA9685 support. Activate the Pi environment and install "
                "adafruit-circuitpython-pca9685."
            ) from error

        try:
            self._pca = PCA9685(
                busio.I2C(board.SCL, board.SDA),
                address=I2C_ADDRESS,
            )
            self._pca.frequency = PWM_FREQUENCY
            self._channels = [self._pca.channels[channel] for channel in ESC_CHANNELS]
            self._current = list(MOTOR_STOP_US)
            self._target = list(MOTOR_STOP_US)
            self._ramp_step_us = ramp_step_us
            self._closed = False
            self.stop(immediate=True)
        except BaseException:
            if self._pca is not None:
                try:
                    if hasattr(self, "_channels"):
                        self._current = list(MOTOR_STOP_US)
                        self._write()
                except BaseException:
                    pass
                try:
                    self._pca.deinit()
                except BaseException:
                    pass
            self._closed = True
            raise

    def arm(self, arm_seconds: float) -> None:
        """Hold verified stop pulses while the ESCs complete startup."""
        print(
            f"Arming ESCs at the supplied {MOTOR_STOP_US[0]} us stop pulse "
            f"for {arm_seconds:.1f} seconds..."
        )
        time.sleep(arm_seconds)

    @staticmethod
    def _pulse_to_duty(pulse_us: int) -> int:
        pulse_us = max(900, min(2100, int(pulse_us)))
        return int((pulse_us * PWM_FREQUENCY * 65535) / 1_000_000)

    def _write(self) -> None:
        with self._io_lock:
            first_error: Exception | None = None
            for channel, pulse_us in zip(self._channels, self._current):
                try:
                    channel.duty_cycle = self._pulse_to_duty(pulse_us)
                except Exception as error:
                    # Still try to stop the other motors if one channel fails.
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error

    def _cancel_motion_watchdog_locked(self) -> None:
        timer = self._motion_stop_timer
        self._motion_stop_timer = None
        self._motion_deadline = None
        if timer is not None:
            timer.cancel()

    def _motion_deadline_expired(self, expected_deadline: float) -> None:
        """Stop independently if camera capture cannot service the main loop."""
        with self._io_lock:
            if self._closed or self._motion_deadline != expected_deadline:
                return
            remaining = expected_deadline - time.monotonic()
            if remaining > 0.001:
                # Timers are normally late, but preserve the absolute deadline
                # if a platform happens to wake this callback slightly early.
                timer = threading.Timer(
                    remaining,
                    self._motion_deadline_expired,
                    args=(expected_deadline,),
                )
                timer.daemon = True
                self._motion_stop_timer = timer
                timer.start()
                return

            self._motion_stop_timer = None
            self._motion_deadline = None
            self._target = list(MOTOR_STOP_US)
            self._current = list(MOTOR_STOP_US)
            stop_error: Exception | None = None
            for _attempt in range(2):
                try:
                    self._write()
                    stop_error = None
                    break
                except Exception as error:
                    stop_error = error
            if stop_error is not None:
                self._watchdog_error = stop_error
                print(
                    f"WARNING: timed ESC stop write failed: {stop_error}",
                    file=sys.stderr,
                )

    def _arm_motion_watchdog_locked(self, deadline: float) -> None:
        if (
            self._motion_deadline == deadline
            and self._motion_stop_timer is not None
        ):
            return
        self._cancel_motion_watchdog_locked()
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            self._target = list(MOTOR_STOP_US)
            self._current = list(MOTOR_STOP_US)
            self._write()
            return
        self._motion_deadline = deadline
        timer = threading.Timer(
            remaining,
            self._motion_deadline_expired,
            args=(deadline,),
        )
        timer.daemon = True
        self._motion_stop_timer = timer
        timer.start()

    @staticmethod
    def _pulse_for_throttle(index: int, throttle: float) -> int:
        throttle = max(0.0, min(1.0, throttle))
        if throttle == 0.0:
            return MOTOR_STOP_US[index]
        return round(
            MOTOR_START_US[index]
            + throttle * (MOTOR_MAX_US[index] - MOTOR_START_US[index])
        )

    def command(
        self,
        motor_throttles: tuple[float, float, float, float],
        immediate_zero: bool = False,
        immediate_positive: bool = False,
        stop_deadline: float | None = None,
    ) -> None:
        with self._io_lock:
            if self._watchdog_error is not None:
                raise RuntimeError("The timed ESC stop watchdog failed") from self._watchdog_error
            self._target = [
                self._pulse_for_throttle(index, value)
                for index, value in enumerate(motor_throttles)
            ]
            if immediate_zero or immediate_positive:
                for index, throttle in enumerate(motor_throttles):
                    if immediate_zero and throttle == 0.0:
                        self._current[index] = MOTOR_STOP_US[index]
                    elif immediate_positive and throttle > 0.0:
                        # A timed creep burst starts from the verified minimum
                        # moving pulse. Ramping from stop would consume the
                        # short burst without ever moving the robot.
                        self._current[index] = self._target[index]
            if stop_deadline is None:
                self._cancel_motion_watchdog_locked()
            else:
                self._arm_motion_watchdog_locked(stop_deadline)

    def step(self) -> None:
        with self._io_lock:
            if self._watchdog_error is not None:
                raise RuntimeError("The timed ESC stop watchdog failed") from self._watchdog_error
            if (
                self._motion_deadline is not None
                and time.monotonic() >= self._motion_deadline
            ):
                self._motion_deadline_expired(self._motion_deadline)
                return
            for index, target in enumerate(self._target):
                difference = target - self._current[index]
                difference = max(-self._ramp_step_us, min(self._ramp_step_us, difference))
                self._current[index] += difference
            self._write()

    def stop(self, immediate: bool = False) -> None:
        with self._io_lock:
            self._cancel_motion_watchdog_locked()
            self._target = list(MOTOR_STOP_US)
            if immediate:
                self._current = list(MOTOR_STOP_US)
                self._write()

    def motion_is_active(self) -> bool:
        """Report actual pulse state, honoring an expired stop deadline."""
        with self._io_lock:
            if (
                self._motion_deadline is not None
                and time.monotonic() >= self._motion_deadline
            ):
                self._motion_deadline_expired(self._motion_deadline)
            return any(
                current != stopped
                for current, stopped in zip(self._current, MOTOR_STOP_US)
            )

    def close(self) -> None:
        if self._closed:
            return
        stop_error: Exception | None = None
        deinit_error: Exception | None = None
        try:
            try:
                self.stop(immediate=True)
            except Exception as error:
                stop_error = error
            time.sleep(0.1)
        finally:
            try:
                self._pca.deinit()
            except Exception as error:
                deinit_error = error
            self._closed = True
        if stop_error is not None:
            print(f"WARNING: ESC stop write failed: {stop_error}", file=sys.stderr)
        if deinit_error is not None:
            print(f"WARNING: PCA9685 shutdown failed: {deinit_error}", file=sys.stderr)


@dataclass(frozen=True)
class DriveCommand:
    name: str
    throttles: tuple[float, float, float, float]


@dataclass
class MotionPulseGate:
    """Alternate motion and camera-only pauses using monotonic wall time."""

    move_seconds: float
    pause_seconds: float
    cycle_started_at: float | None = None
    move_deadline: float | None = None

    @property
    def duty_cycle(self) -> float:
        cycle_seconds = self.move_seconds + self.pause_seconds
        return self.move_seconds / cycle_seconds

    def reset(self) -> None:
        self.cycle_started_at = None
        self.move_deadline = None

    def limit(self, command: DriveCommand, now: float) -> DriveCommand:
        """Return all-stop during VIEW and the latest plan during MOVE.

        A fresh cycle deliberately starts with the camera-only pause. Positive
        command changes do not restart the clock, so continually changing
        steering plans cannot accidentally defeat the configured movement duty.
        """
        if not any(command.throttles):
            self.reset()
            return command
        if self.pause_seconds == 0.0:
            self.move_deadline = None
            return command
        if self.cycle_started_at is None or now < self.cycle_started_at:
            self.cycle_started_at = now

        cycle_seconds = self.move_seconds + self.pause_seconds
        elapsed = now - self.cycle_started_at
        cycle_number = int(elapsed // cycle_seconds)
        cycle_start = self.cycle_started_at + cycle_number * cycle_seconds
        cycle_position = elapsed % cycle_seconds
        if cycle_position >= self.pause_seconds:
            self.move_deadline = cycle_start + cycle_seconds
            return command
        self.move_deadline = None
        return DriveCommand(
            f"VIEW PAUSE - NEXT {command.name}",
            (0.0, 0.0, 0.0, 0.0),
        )


def side_command(name: str, right_turn: float, left_turn: float) -> DriveCommand:
    """Build a command using the motor grouping proven by the manual UI."""
    values = [0.0] * 4
    for index in RIGHT_TURN_MOTORS:
        values[index] = right_turn
    for index in LEFT_TURN_MOTORS:
        values[index] = left_turn
    return DriveCommand(name, tuple(values))  # type: ignore[arg-type]


def course_is_complete(navigator: object, args: argparse.Namespace) -> bool:
    """Latch completion once the configured number of cones has been passed."""
    max_cones = getattr(args, "max_cones", None)
    return (
        bool(getattr(navigator, "course_complete", False))
        or (
            isinstance(max_cones, int)
            and max_cones >= 1
            and navigator.cones_passed >= max_cones
        )
    )


def course_complete_feedback(args: argparse.Namespace) -> str:
    return f"COURSE COMPLETE - {args.max_cones} CONES PASSED - MOTORS STOPPED"


def new_preview_navigator(args: argparse.Namespace) -> object:
    """Create paused camera feedback that cannot complete the real course."""
    preview = new_navigator(args)
    preview.max_cones = None
    return preview


def synchronize_preview_course(preview: object, navigator: object) -> None:
    """Keep paused display progress tied to the real autonomous run."""
    preview.cones_passed = navigator.cones_passed
    preview.direction_index = navigator.direction_index


def apply_drive_output(
    drive: FourEscDrive,
    command: DriveCommand,
    args: argparse.Namespace,
    course_complete: bool,
    move_deadline: float | None = None,
) -> bool:
    """Apply one command, bypassing all ramps for every all-motor stop."""
    if course_complete or not any(command.throttles):
        drive.stop(immediate=True)
        return False
    drive.command(
        command.throttles,
        # Deceleration is never ramped. This makes the stopped inside motor
        # pair create a real turn instead of continuing to push forward.
        immediate_zero=True,
        immediate_positive=getattr(args, "creep_pause_seconds", 0.0) > 0.0,
        stop_deadline=move_deadline,
    )
    drive.step()
    motion_check = getattr(drive, "motion_is_active", None)
    return bool(motion_check()) if callable(motion_check) else True


def choose_drive_command(navigator: object, frame_width: int, args: argparse.Namespace) -> DriveCommand:
    """Drive straight to the turn distance, then use one gentle turn speed."""
    stop = side_command("STOP", 0.0, 0.0)
    target = navigator.current_target

    if course_is_complete(navigator, args):
        return side_command("STOP - COURSE COMPLETE", 0.0, 0.0)
    if navigator.phase == "CALIBRATION REQUIRED":
        return stop
    if navigator.close_cone_hazard:
        return side_command("STOP - CLOSE CONE", 0.0, 0.0)

    clearance_direction = getattr(
        navigator,
        "passed_cone_clearance_direction",
        None,
    )
    clearance = bool(
        getattr(navigator, "passed_cone_too_close", False)
        and clearance_direction in {"LEFT", "RIGHT"}
    )
    # Countersteering and finding the next cone require the newly alternated turn.
    if clearance and clearance_direction in {"LEFT", "RIGHT"}:
        direction = clearance_direction
    elif navigator.countersteer_remaining > 0 or (
        target is None and navigator.awaiting_new_cone
    ):
        direction = navigator.direction
    elif target is None:
        return stop
    elif navigator.phase in {"TURNING", "PASSING"}:
        direction = navigator.direction
    else:
        # Do not steer early based only on the cone's image position. The
        # navigator enters TURNING at the configured distance, so approach is
        # a slow straight-ahead motion.
        if getattr(args, "turn_test_mode", False):
            return side_command("APPROACH - NO TURN", 0.0, 0.0)
        return side_command("FORWARD", args.cruise_throttle, args.cruise_throttle)

    # There is deliberately no hard-turn branch. Turning, counterturning, and
    # searching all use this same low forward-only outside-pair output; the
    # inside pair is stopped and is never commanded in reverse.
    inside = 0.0 if getattr(args, "turn_test_mode", False) else args.turn_inside_throttle
    outside = args.turn_outside_throttle
    name = f"CLEAR {direction}" if clearance else direction
    if direction == "RIGHT":
        return side_command(name, outside, inside)
    return side_command(name, inside, outside)


def enforce_next_cone_timeout(
    command: DriveCommand,
    navigator: object,
    paused: bool,
    started_at: float | None,
    now: float,
    timeout_seconds: float,
) -> tuple[DriveCommand, float | None]:
    """Bound all post-pass searching, including the countersteer interval."""
    searching = (
        not paused
        and bool(getattr(navigator, "awaiting_new_cone", False))
        and not bool(getattr(navigator, "close_cone_hazard", False))
    )
    if not searching:
        return command, None
    if started_at is None or now < started_at:
        started_at = now
    if now - started_at >= timeout_seconds:
        return side_command("STOP - NEXT CONE NOT FOUND", 0.0, 0.0), started_at
    return command, started_at


def turn_test_banner(
    command: DriveCommand,
    paused: bool,
) -> tuple[str, str, tuple[int, int, int]]:
    """Return an unmistakable raised-wheel direction and motor-group label."""
    if command.name == "COURSE COMPLETE":
        return (
            "COURSE COMPLETE - ALL STOP",
            "ALL REQUIRED CONES PASSED  |  PRESS R TO RESET",
            (90, 235, 90),
        )
    if paused:
        return (
            "TURN TEST - PAUSED",
            "LEFT = M3-4  |  RIGHT = M1-2  |  FAR CENTER = ALL STOP",
            (80, 230, 255),
        )
    if command.name.startswith("VIEW PAUSE - NEXT "):
        next_command = command.name.removeprefix("VIEW PAUSE - NEXT ")
        return (
            "CREEP: CAMERA VIEW PAUSE - ALL STOP",
            f"NEXT MOTION COMMAND: {next_command}",
            (80, 230, 255),
        )
    if command.name in {"LEFT", "CLEAR LEFT"}:
        return (
            "<<<  TURN TEST: LEFT",
            "COMMANDED: MOTORS 3-4 RUN  |  MOTORS 1-2 STOP",
            (255, 220, 0),
        )
    if command.name in {"RIGHT", "CLEAR RIGHT"}:
        return (
            "TURN TEST: RIGHT  >>>",
            "COMMANDED: MOTORS 1-2 RUN  |  MOTORS 3-4 STOP",
            (0, 165, 255),
        )
    if command.name == "APPROACH - NO TURN":
        return (
            "APPROACH PHASE - ALL STOP",
            "BRING THE CONE TO THE TURN DISTANCE TO TEST THE PLANNED TURN",
            (90, 235, 90),
        )
    if command.name == "FORWARD":
        return "FORWARD", "ALL FOUR MOTORS COMMANDED", (90, 235, 90)
    return "STOP", "ALL FOUR MOTORS STOP", (80, 80, 255)


def draw_autonomous_controls(
    dashboard: object,
    paused: bool,
    command: DriveCommand,
    turn_test_mode: bool,
    cones_passed: int,
    max_cones: int,
    course_complete: bool,
    creep_duty_cycle: float,
) -> None:
    """Replace the detector-only footer with autonomous drive controls."""
    if turn_test_mode:
        title, detail, color = turn_test_banner(command, paused)
        cv2.rectangle(
            dashboard,
            (0, 0),
            (dashboard.shape[1], 106),
            (15, 15, 15),
            -1,
        )
        title_scale = min(1.25, max(0.75, dashboard.shape[1] / 1050.0))
        title_size = cv2.getTextSize(
            title,
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            4,
        )[0]
        title_x = max(12, (dashboard.shape[1] - title_size[0]) // 2)
        cv2.putText(
            dashboard,
            title,
            (title_x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (0, 0, 0),
            8,
        )
        cv2.putText(
            dashboard,
            title,
            (title_x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            color,
            4,
        )
        detail_scale = min(0.58, max(0.38, dashboard.shape[1] / 2400.0))
        detail_size = cv2.getTextSize(
            detail,
            cv2.FONT_HERSHEY_SIMPLEX,
            detail_scale,
            2,
        )[0]
        detail_x = max(12, (dashboard.shape[1] - detail_size[0]) // 2)
        cv2.putText(
            dashboard,
            detail,
            (detail_x, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            detail_scale,
            (225, 225, 225),
            2,
        )
        warning = "WHEELS RAISED ONLY   |   SPACE/S = IMMEDIATE STOP"
        warning_scale = min(0.48, max(0.34, dashboard.shape[1] / 2800.0))
        warning_size = cv2.getTextSize(
            warning,
            cv2.FONT_HERSHEY_SIMPLEX,
            warning_scale,
            1,
        )[0]
        warning_x = max(12, (dashboard.shape[1] - warning_size[0]) // 2)
        cv2.putText(
            dashboard,
            warning,
            (warning_x, 99),
            cv2.FONT_HERSHEY_SIMPLEX,
            warning_scale,
            (80, 80, 255),
            1,
        )

    cv2.rectangle(
        dashboard,
        (0, 108),
        (dashboard.shape[1], 137),
        (28, 28, 28),
        -1,
    )
    if course_complete:
        footer = (
            f"COMPLETE {cones_passed}/{max_cones}   |   ALL MOTORS STOPPED   |   "
            "R = RESET   Q/ESC = QUIT"
        )
        footer_color = (90, 235, 90)
    elif command.name.startswith("VIEW PAUSE - NEXT "):
        next_command = command.name.removeprefix("VIEW PAUSE - NEXT ")
        footer = (
            f"AUTO {cones_passed}/{max_cones}   |   CREEP VIEW PAUSE {creep_duty_cycle:.0%}   |   "
            f"ALL MOTORS STOPPED   |   NEXT: {next_command}   |   S = OPERATOR STOP"
        )
        footer_color = (80, 230, 255)
    elif paused:
        footer = (
            f"OPERATOR PAUSED {cones_passed}/{max_cones}   |   ALL MOTORS STOPPED   |   "
            "G = GO   R = RESET   Q/ESC = QUIT"
        )
        footer_color = (80, 230, 255)
    elif not any(command.throttles):
        footer = (
            f"AUTO STOP {cones_passed}/{max_cones}   |   ALL MOTORS STOPPED   |   "
            f"{command.name}   |   S = OPERATOR STOP   R = RESET"
        )
        footer_color = (80, 80, 255)
    else:
        footer = (
            f"AUTO MOVE {cones_passed}/{max_cones}   |   CREEP {creep_duty_cycle:.0%}   |   "
            "SPACE/S = OPERATOR STOP   R = RESET   Q/ESC = QUIT"
        )
        footer_color = (120, 255, 120)
    footer_scale = 0.48
    footer_width = cv2.getTextSize(
        footer,
        cv2.FONT_HERSHEY_SIMPLEX,
        footer_scale,
        1,
    )[0][0]
    if footer_width > dashboard.shape[1] - 32:
        footer_scale = max(
            0.32,
            footer_scale * (dashboard.shape[1] - 32) / footer_width,
        )
    cv2.putText(
        dashboard,
        footer,
        (16, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        footer_scale,
        footer_color,
        1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous red-cone slalom driver")
    parser.add_argument("--backend", choices=("auto", "picamera2", "opencv"), default="auto")
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
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the live dashboard; autonomy starts immediately",
    )
    parser.add_argument(
        "--turn-test-mode",
        action="store_true",
        help=(
            "raised-wheel test only: show a large LEFT/RIGHT banner and stop "
            "the inside motor pair during turns"
        ),
    )
    parser.add_argument("--cone-height-cm", type=float, default=30.5)
    parser.add_argument("--calibration-distance-cm", type=float, default=100.0)
    pi_calibration_file = Path(__file__).with_name("pi_cone_camera_calibration.json")
    parser.add_argument("--calibration-file", type=Path, default=pi_calibration_file)
    parser.add_argument(
        "--turn-start-cm",
        type=float,
        default=160.0,
        help="begin the slow slalom turn at this calibrated cone distance",
    )
    parser.add_argument(
        "--turn-start-height-ratio",
        type=float,
        default=0.30,
        help=(
            "close-cone safety threshold as a fraction of image height; "
            "normal turns begin from calibrated distance only"
        ),
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
    parser.add_argument(
        "--countersteer-frames",
        type=int,
        default=12,
        help="applied-motion frames after a pass; camera-view pauses do not count",
    )
    parser.add_argument(
        "--max-cones",
        type=int,
        default=3,
        help="stop and remain stopped after this many confirmed cone passes",
    )
    parser.add_argument("--cruise-throttle", type=float, default=0.003)
    parser.add_argument(
        "--turn-outside-throttle",
        type=float,
        default=0.015,
        help=(
            "forward-only outside-pair turn power; the higher default supplies "
            "enough torque for two motors to drag the stopped inside wheels"
        ),
    )
    parser.add_argument("--turn-inside-throttle", type=float, default=0.0)
    parser.add_argument("--arm-seconds", type=float, default=5.0)
    parser.add_argument("--ramp-step-us", type=int, default=3)
    parser.add_argument(
        "--creep-move-seconds",
        type=float,
        default=0.20,
        help="seconds of minimum-pulse movement in each creep cycle",
    )
    parser.add_argument(
        "--creep-pause-seconds",
        type=float,
        default=0.30,
        help="all-stop camera observation time before and between movement bursts",
    )
    parser.add_argument("--camera-loss-frames", type=int, default=3)
    parser.add_argument(
        "--search-timeout-seconds",
        type=float,
        default=4.0,
        help=(
            "wall-clock seconds after a pass to find the next cone, including "
            "countersteering"
        ),
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="required acknowledgement that the robot may move",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_camera_args(args)
    for name in ("cruise_throttle", "turn_outside_throttle", "turn_inside_throttle"):
        if not 0.0 <= getattr(args, name) <= 0.25:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 0.25")
    if not (
        args.turn_outside_throttle >= args.cruise_throttle
        and args.turn_outside_throttle > args.turn_inside_throttle
    ):
        raise SystemExit(
            "Throttle settings must satisfy: outside >= cruise and outside > inside"
        )
    if (
        args.ramp_step_us < 1
        or args.camera_loss_frames < 1
        or args.search_timeout_seconds <= 0
        or args.max_cones < 1
        or args.creep_move_seconds <= 0
    ):
        raise SystemExit(
            "Ramp step, camera-loss frames, search timeout, max cones, and "
            "creep move time must be positive"
        )
    if args.creep_pause_seconds < 0:
        raise SystemExit("Creep pause time cannot be negative")
    if not args.drive:
        raise SystemExit("Refusing to move: rerun with --drive after raising the wheels and checking the kill switch.")
    if args.turn_test_mode and args.headless:
        raise SystemExit("Turn-test mode requires the live dashboard; remove --headless.")
    if args.turn_test_mode and args.turn_outside_throttle <= 0.0:
        raise SystemExit("Turn-test mode needs --turn-outside-throttle greater than zero.")

    drive: FourEscDrive | None = None
    camera = None
    try:
        # Convert termination signals into a Python exit before touching the
        # motor controller so partial initialization can run its cleanup path.
        signal.signal(signal.SIGTERM, lambda *_: raise_system_exit())
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, lambda *_: raise_system_exit())

        # When motor power is already connected, make the first hardware
        # action a stop command. Camera startup and calibration loading happen
        # only after all four ESC channels are at their verified stop pulses.
        drive = FourEscDrive(args.ramp_step_us)
        atexit.register(drive.close)
        drive.arm(args.arm_seconds)

        calibration = CameraCalibration.load(args.calibration_file)
        if calibration is None:
            raise SystemExit(
                "No Pi slalom calibration found. Run cone_detector.py with --calibrate "
                f"--calibration-file {args.calibration_file} before connecting motor power."
            )
        if (
            calibration.frame_width != args.width
            or calibration.frame_height != args.height
            or abs(calibration.cone_height_cm - args.cone_height_cm) > 0.05
        ):
            raise SystemExit(
                "Calibration does not match the requested resolution or cone height. "
                "Recalibrate this Pi camera with the same width, height, and cone-height options."
            )

        camera = create_camera(args)
        navigator = new_navigator(args)
        preview_navigator = new_preview_navigator(args)
        motion_gate = MotionPulseGate(
            move_seconds=args.creep_move_seconds,
            pause_seconds=args.creep_pause_seconds,
        )
        lost_frames = 0
        next_cone_search_started_at: float | None = None
        previous_status = ""
        paused = not args.headless
        has_started = args.headless
        motion_applied_last_frame = False
        print(f"Autonomous slalom camera active using {camera.name}.")
        if args.headless:
            print("Headless autonomy starts immediately. Ctrl+C stops all motors.")
        else:
            print("Live dashboard starts PAUSED. G=go, Space/S=stop, R=reset, Q=quit.")
        print(
            f"Course plan: drive and alternate around {args.max_cones} cones, "
            "then stop and wait for R."
        )
        print(
            "Minimum-speed pulses: "
            f"forward {FourEscDrive._pulse_for_throttle(0, args.cruise_throttle)} us, "
            f"turn outside {FourEscDrive._pulse_for_throttle(0, args.turn_outside_throttle)} us, "
            f"turn inside {FourEscDrive._pulse_for_throttle(0, args.turn_inside_throttle)} us."
        )
        camera_offset_cm = args.camera_from_left_cm - args.robot_width_cm / 2.0
        if camera_offset_cm < 0.0:
            camera_position = f"{abs(camera_offset_cm):.2f} cm left of vehicle center"
        elif camera_offset_cm > 0.0:
            camera_position = f"{camera_offset_cm:.2f} cm right of vehicle center"
        else:
            camera_position = "on the vehicle centerline"
        print(
            f"Camera geometry: {args.robot_width_cm:.2f} cm robot width, camera "
            f"{args.camera_from_left_cm:.2f} cm from the left "
            f"({camera_position})."
        )
        print(
            f"Creep starts with {args.creep_pause_seconds:.2f} s ALL-STOP CAMERA VIEW, "
            f"then alternates {args.creep_move_seconds:.2f} s MOVE / "
            f"{args.creep_pause_seconds:.2f} s VIEW "
            f"({motion_gate.duty_cycle:.0%} movement duty)."
        )
        if args.turn_test_mode:
            print(
                "TURN TEST MODE: LEFT runs motors 3-4, RIGHT runs motors 1-2, "
                "and the opposite pair stops. Raised wheels only."
            )

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                lost_frames += 1
                # Do not coast on the last vision command while retrying the camera.
                drive.stop(immediate=True)
                motion_gate.reset()
                motion_applied_last_frame = False
                if lost_frames >= args.camera_loss_frames:
                    raise RuntimeError("Camera frames lost; motors stopped")
                continue

            lost_frames = 0
            detections = detect_cones(frame, min_area=args.min_area)
            course_complete = course_is_complete(navigator, args)
            view_navigator = (
                navigator
                if course_complete
                else preview_navigator if paused else navigator
            )
            if course_complete:
                feedback = course_complete_feedback(args)
            else:
                feedback = view_navigator.update(
                    detections,
                    calibration,
                    frame.shape,
                    motion_applied=(
                        motion_applied_last_frame if not paused else False
                    ),
                )
                if paused:
                    # Preview tracking may react to cones moved by hand, but it
                    # must never display progress or direction from a run that
                    # did not actually happen.
                    synchronize_preview_course(preview_navigator, navigator)
                course_complete = course_is_complete(navigator, args)
                if course_complete:
                    feedback = course_complete_feedback(args)
            if course_complete:
                command = side_command("COURSE COMPLETE", 0.0, 0.0)
                next_cone_search_started_at = None
            else:
                command = choose_drive_command(view_navigator, frame.shape[1], args)
                command, next_cone_search_started_at = enforce_next_cone_timeout(
                    command,
                    view_navigator,
                    paused,
                    next_cone_search_started_at,
                    time.monotonic(),
                    args.search_timeout_seconds,
                )

                if paused:
                    command = side_command("PAUSED", 0.0, 0.0)

            planned_command = command
            command = motion_gate.limit(planned_command, time.monotonic())
            motion_applied_last_frame = apply_drive_output(
                drive,
                command,
                args,
                course_complete,
                move_deadline=motion_gate.move_deadline,
            )
            if any(command.throttles) and not motion_applied_last_frame:
                command = DriveCommand(
                    f"VIEW PAUSE - NEXT {planned_command.name}",
                    (0.0, 0.0, 0.0, 0.0),
                )

            status = f"NAV PLAN {planned_command.name}: {feedback}"
            if status != previous_status:
                print(status)
                previous_status = status

            if not args.headless:
                mask = make_red_mask(frame)
                display_detections = detections_for_dashboard(
                    detections,
                    view_navigator.current_target,
                )
                if course_complete:
                    visual_feedback = (
                        f"COURSE COMPLETE: {args.max_cones} CONES - STOPPED (R = RESET)"
                    )
                elif paused:
                    visual_feedback = "PAUSED - PRESS G TO START AUTONOMOUS DRIVE"
                elif command.name.startswith("VIEW PAUSE - NEXT "):
                    next_command = command.name.removeprefix("VIEW PAUSE - NEXT ")
                    visual_feedback = (
                        f"ALL MOTORS STOPPED - CAMERA VIEW | NEXT: {next_command} | "
                        f"CONES {view_navigator.cones_passed}/{args.max_cones} | {feedback}"
                    )
                else:
                    visual_feedback = (
                        f"MOTORS {command.name} | CONES "
                        f"{view_navigator.cones_passed}/{args.max_cones} | {feedback}"
                    )
                dashboard = make_dashboard(
                    frame,
                    mask,
                    display_detections,
                    view_navigator,
                    visual_feedback,
                    view_navigator.smoothed_distance_cm,
                    calibration,
                    args.calibration_distance_cm,
                    display_width=args.display_width,
                )
                draw_autonomous_controls(
                    dashboard,
                    paused,
                    command,
                    args.turn_test_mode,
                    navigator.cones_passed,
                    args.max_cones,
                    course_complete,
                    motion_gate.duty_cycle,
                )
                cv2.imshow(WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF
                ui_framework = (
                    cv2.currentUIFramework().upper()
                    if hasattr(cv2, "currentUIFramework")
                    else ""
                )
                if ui_framework == "WAYLAND":
                    window_visible = 1.0
                else:
                    try:
                        window_visible = cv2.getWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_VISIBLE
                        )
                    except cv2.error:
                        window_visible = 1.0
                if window_visible == 0:
                    drive.stop(immediate=True)
                    break
                if key in (ord("q"), ord("Q"), 27):
                    drive.stop(immediate=True)
                    break
                if key in (ord(" "), ord("s"), ord("S")):
                    paused = True
                    preview_navigator = new_preview_navigator(args)
                    next_cone_search_started_at = None
                    drive.stop(immediate=True)
                    motion_gate.reset()
                    motion_applied_last_frame = False
                elif key in (ord("g"), ord("G")):
                    if course_complete:
                        # Completion is latched. Only R may start a new course.
                        drive.stop(immediate=True)
                        motion_gate.reset()
                        motion_applied_last_frame = False
                    else:
                        if not has_started:
                            navigator = new_navigator(args)
                            has_started = True
                        next_cone_search_started_at = None
                        paused = False
                elif key in (ord("r"), ord("R")):
                    navigator = new_navigator(args)
                    preview_navigator = new_preview_navigator(args)
                    has_started = False
                    next_cone_search_started_at = None
                    paused = True
                    drive.stop(immediate=True)
                    motion_gate.reset()
                    motion_applied_last_frame = False
    except KeyboardInterrupt:
        print("Stopped by operator.")
    finally:
        if drive is not None:
            drive.close()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    main()
