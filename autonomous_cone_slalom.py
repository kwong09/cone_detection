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
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from cone_detection_iteration_1 import detect_cones, make_red_mask
from cone_detection_iteration_3 import CameraCalibration, make_dashboard
from cone_detection_iteration_5_pi import (
    create_camera,
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

    def __init__(self, arm_seconds: float, ramp_step_us: int) -> None:
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError as error:
            raise RuntimeError(
                "Missing PCA9685 support. Activate the Pi environment and install "
                "adafruit-circuitpython-pca9685."
            ) from error

        self._pca = PCA9685(busio.I2C(board.SCL, board.SDA), address=I2C_ADDRESS)
        self._pca.frequency = PWM_FREQUENCY
        self._channels = [self._pca.channels[channel] for channel in ESC_CHANNELS]
        self._current = list(MOTOR_STOP_US)
        self._target = list(MOTOR_STOP_US)
        self._ramp_step_us = ramp_step_us
        self._closed = False
        self.stop(immediate=True)
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
        first_error: Exception | None = None
        for channel, pulse_us in zip(self._channels, self._current):
            try:
                channel.duty_cycle = self._pulse_to_duty(pulse_us)
            except Exception as error:
                # Still try to stop the other motors if one PCA channel fails.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    @staticmethod
    def _pulse_for_throttle(index: int, throttle: float) -> int:
        throttle = max(0.0, min(1.0, throttle))
        if throttle == 0.0:
            return MOTOR_STOP_US[index]
        return round(
            MOTOR_START_US[index]
            + throttle * (MOTOR_MAX_US[index] - MOTOR_START_US[index])
        )

    def command(self, motor_throttles: tuple[float, float, float, float]) -> None:
        self._target = [
            self._pulse_for_throttle(index, value)
            for index, value in enumerate(motor_throttles)
        ]

    def step(self) -> None:
        for index, target in enumerate(self._target):
            difference = target - self._current[index]
            difference = max(-self._ramp_step_us, min(self._ramp_step_us, difference))
            self._current[index] += difference
        self._write()

    def stop(self, immediate: bool = False) -> None:
        self._target = list(MOTOR_STOP_US)
        if immediate:
            self._current = list(MOTOR_STOP_US)
            self._write()

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


def side_command(name: str, right_turn: float, left_turn: float) -> DriveCommand:
    """Build a command using the motor grouping proven by the manual UI."""
    values = [0.0] * 4
    for index in RIGHT_TURN_MOTORS:
        values[index] = right_turn
    for index in LEFT_TURN_MOTORS:
        values[index] = left_turn
    return DriveCommand(name, tuple(values))  # type: ignore[arg-type]


def choose_drive_command(navigator: object, frame_width: int, args: argparse.Namespace) -> DriveCommand:
    """Convert navigator state into conservative differential motor output."""
    stop = side_command("STOP", 0.0, 0.0)
    target = navigator.current_target

    if navigator.phase == "CALIBRATION REQUIRED":
        return stop

    # Countersteering and finding the next cone require the newly alternated turn.
    if navigator.countersteer_remaining > 0 or (
        target is None and navigator.awaiting_new_cone
    ):
        direction = navigator.direction
    elif target is None:
        return stop
    elif navigator.phase in {"TURNING", "PASSING"}:
        direction = navigator.direction
    else:
        x_ratio = target.center[0] / frame_width
        if x_ratio < 0.44:
            direction = "LEFT"
        elif x_ratio > 0.56:
            direction = "RIGHT"
        else:
            return side_command("FORWARD", args.cruise_throttle, args.cruise_throttle)

    hard = (
        (
            getattr(navigator.current_target, "cropped", False)
            or (
                navigator.smoothed_distance_cm is not None
                and navigator.smoothed_distance_cm <= args.hard_turn_cm
            )
        )
        and not navigator.clearance_seen
    )
    inside = args.hard_inside_throttle if hard else args.turn_inside_throttle
    outside = args.turn_outside_throttle
    if direction == "RIGHT":
        return side_command("HARD RIGHT" if hard else "RIGHT", outside, inside)
    return side_command("HARD LEFT" if hard else "LEFT", inside, outside)


def draw_autonomous_controls(dashboard: object, paused: bool) -> None:
    """Replace the detector-only footer with autonomous drive controls."""
    cv2.rectangle(
        dashboard,
        (0, 108),
        (dashboard.shape[1], 137),
        (28, 28, 28),
        -1,
    )
    state = "PAUSED" if paused else "DRIVING"
    cv2.putText(
        dashboard,
        f"{state}   |   G = GO   SPACE/S = STOP   R = RESET + STOP   Q/ESC = QUIT",
        (16, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 230, 255) if paused else (120, 255, 120),
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
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the live dashboard; autonomy starts immediately",
    )
    parser.add_argument("--cone-height-cm", type=float, default=30.5)
    parser.add_argument("--calibration-distance-cm", type=float, default=100.0)
    pi_calibration_file = Path(__file__).with_name("pi_cone_camera_calibration.json")
    parser.add_argument("--calibration-file", type=Path, default=pi_calibration_file)
    parser.add_argument("--turn-start-cm", type=float, default=130.0)
    parser.add_argument("--hard-turn-cm", type=float, default=80.0)
    parser.add_argument("--pass-distance-cm", type=float, default=60.0)
    parser.add_argument("--countersteer-frames", type=int, default=12)
    parser.add_argument("--cruise-throttle", type=float, default=0.16)
    parser.add_argument("--turn-outside-throttle", type=float, default=0.18)
    parser.add_argument("--turn-inside-throttle", type=float, default=0.07)
    parser.add_argument("--hard-inside-throttle", type=float, default=0.0)
    parser.add_argument("--arm-seconds", type=float, default=5.0)
    parser.add_argument("--ramp-step-us", type=int, default=25)
    parser.add_argument("--camera-loss-frames", type=int, default=3)
    parser.add_argument(
        "--search-timeout-seconds",
        type=float,
        default=2.0,
        help="stop a blind search if the next cone is not found in this time",
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
    for name in ("cruise_throttle", "turn_outside_throttle", "turn_inside_throttle", "hard_inside_throttle"):
        if not 0.0 <= getattr(args, name) <= 0.25:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 0.25")
    if (
        args.ramp_step_us < 1
        or args.camera_loss_frames < 1
        or args.search_timeout_seconds <= 0
    ):
        raise SystemExit("Ramp step, camera-loss frames, and search timeout must be positive")
    if not args.drive:
        raise SystemExit("Refusing to move: rerun with --drive after raising the wheels and checking the kill switch.")

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
    drive: FourEscDrive | None = None
    try:
        drive = FourEscDrive(args.arm_seconds, args.ramp_step_us)
        atexit.register(drive.close)
        signal.signal(signal.SIGTERM, lambda *_: raise_system_exit())
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, lambda *_: raise_system_exit())
        navigator = new_navigator(args)
        preview_navigator = new_navigator(args)
        lost_frames = 0
        blind_search_started_at: float | None = None
        previous_status = ""
        paused = not args.headless
        has_started = args.headless
        print(f"Autonomous slalom camera active using {camera.name}.")
        if args.headless:
            print("Headless autonomy starts immediately. Ctrl+C stops all motors.")
        else:
            print("Live dashboard starts PAUSED. G=go, Space/S=stop, R=reset, Q=quit.")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                lost_frames += 1
                # Do not coast on the last vision command while retrying the camera.
                drive.stop(immediate=True)
                if lost_frames >= args.camera_loss_frames:
                    raise RuntimeError("Camera frames lost; motors stopped")
                continue

            lost_frames = 0
            detections = detect_cones(frame, min_area=args.min_area)
            view_navigator = preview_navigator if paused else navigator
            feedback = view_navigator.update(detections, calibration, frame.shape)
            command = choose_drive_command(view_navigator, frame.shape[1], args)

            blind_searching = (
                not paused
                and view_navigator.current_target is None
                and view_navigator.awaiting_new_cone
                and view_navigator.countersteer_remaining == 0
            )
            if blind_searching:
                if blind_search_started_at is None:
                    blind_search_started_at = time.monotonic()
                if (
                    time.monotonic() - blind_search_started_at
                    >= args.search_timeout_seconds
                ):
                    command = side_command("STOP - NEXT CONE NOT FOUND", 0.0, 0.0)
            else:
                blind_search_started_at = None

            if paused:
                command = side_command("PAUSED", 0.0, 0.0)

            drive.command(command.throttles)
            drive.step()

            status = f"{command.name}: {feedback}"
            if status != previous_status:
                print(status)
                previous_status = status

            if not args.headless:
                mask = make_red_mask(frame)
                target_detections = (
                    [view_navigator.current_target]
                    if view_navigator.current_target is not None
                    else []
                )
                visual_feedback = (
                    "PAUSED - PRESS G TO START AUTONOMOUS DRIVE"
                    if paused
                    else f"MOTORS {command.name} | {feedback}"
                )
                dashboard = make_dashboard(
                    frame,
                    mask,
                    target_detections,
                    view_navigator,
                    visual_feedback,
                    view_navigator.smoothed_distance_cm,
                    calibration,
                    args.calibration_distance_cm,
                    display_width=args.display_width,
                )
                draw_autonomous_controls(dashboard, paused)
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
                    preview_navigator = new_navigator(args)
                    blind_search_started_at = None
                    drive.stop(immediate=True)
                elif key in (ord("g"), ord("G")):
                    if not has_started:
                        navigator = new_navigator(args)
                        has_started = True
                    blind_search_started_at = None
                    paused = False
                elif key in (ord("r"), ord("R")):
                    navigator = new_navigator(args)
                    preview_navigator = new_navigator(args)
                    has_started = False
                    blind_search_started_at = None
                    paused = True
                    drive.stop(immediate=True)
    except KeyboardInterrupt:
        print("Stopped by operator.")
    finally:
        if drive is not None:
            drive.close()
        camera.close()
        cv2.destroyAllWindows()


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    main()
