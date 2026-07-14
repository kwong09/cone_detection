#!/usr/bin/env python3
"""Keyboard control for four Maxynos 45 A bidirectional brushless ESCs.

Hardware:
  Raspberry Pi 5 -> I2C PCA9685 -> four Maxynos ESC signal inputs
  Each ESC -> one MAXYNOS THOR-2826 brushless motor

Lift all wheels, fit a physical motor-power disconnect, and verify the motor
power system before applying power. The keyboard is not an emergency stop.
"""

from __future__ import annotations

import atexit
import select
import signal
import sys
import termios
import time
import tty

import board
import busio
from adafruit_pca9685 import PCA9685


# PCA9685 channel assignments. Change these only if your wiring differs.
FRONT_LEFT = 0
REAR_LEFT = 1
FRONT_RIGHT = 2
REAR_RIGHT = 3

LEFT_CHANNELS = (FRONT_LEFT, REAR_LEFT)
RIGHT_CHANNELS = (FRONT_RIGHT, REAR_RIGHT)


class FourWheelDrive:
    """Drive a four-motor skid-steer robot with signed throttle commands."""

    PWM_FREQUENCY_HZ = 50

    # Maxynos 45 A ESC normal-PWM mapping.
    FULL_REVERSE_PULSE_US = 1000
    NEUTRAL_PULSE_US = 1500
    FULL_FORWARD_PULSE_US = 2000

    # First-test ceiling: 20% of either half of the command range.
    MAX_OUTPUT = 0.20

    # Ramp gently and pause at neutral before changing direction.
    RAMP_STEP = 0.02
    CONTROL_INTERVAL_S = 0.05
    REVERSAL_NEUTRAL_S = 0.30
    ARM_TIME_S = 4.0

    # Loss of keyboard/SSH input commands neutral within this time.
    DRIVE_WATCHDOG_S = 0.80

    def __init__(self) -> None:
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=0x40)
        self.pca.frequency = self.PWM_FREQUENCY_HZ
        self._left = 0.0
        self._right = 0.0
        self._left_reverse_allowed_at = 0.0
        self._right_reverse_allowed_at = 0.0
        self._closed = False

        self.neutral()
        print(
            f"Sending {self.NEUTRAL_PULSE_US} us neutral pulses for "
            f"{self.ARM_TIME_S:.0f} seconds. Keep all wheels raised..."
        )
        time.sleep(self.ARM_TIME_S)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _opposite_sign(first: float, second: float) -> bool:
        return (first > 0.0 > second) or (first < 0.0 < second)

    def _throttle_to_pulse_us(self, throttle: float) -> int:
        """Convert signed throttle (-1..1) to a Maxynos ESC PWM pulse."""
        throttle = self._clamp(throttle, -1.0, 1.0)
        if throttle >= 0.0:
            return round(
                self.NEUTRAL_PULSE_US
                + throttle
                * (self.FULL_FORWARD_PULSE_US - self.NEUTRAL_PULSE_US)
            )
        return round(
            self.NEUTRAL_PULSE_US
            + throttle
            * (self.NEUTRAL_PULSE_US - self.FULL_REVERSE_PULSE_US)
        )

    def _write_channel(self, channel: int, throttle: float) -> None:
        pulse_us = self._throttle_to_pulse_us(throttle)
        period_us = 1_000_000 / self.PWM_FREQUENCY_HZ
        duty_cycle = round((pulse_us / period_us) * 0xFFFF)
        self.pca.channels[channel].duty_cycle = duty_cycle

    def _write_sides(self) -> None:
        for channel in LEFT_CHANNELS:
            self._write_channel(channel, self._left)
        for channel in RIGHT_CHANNELS:
            self._write_channel(channel, self._right)

    def _step_side(
        self,
        current: float,
        requested: float,
        reverse_allowed_at: float,
        now: float,
    ) -> tuple[float, float]:
        target = self._clamp(requested, -1.0, 1.0) * self.MAX_OUTPUT

        # Ramp to neutral before accepting a command in the other direction.
        if self._opposite_sign(current, target):
            target = 0.0

        error = target - current
        updated = current + self._clamp(error, -self.RAMP_STEP, self.RAMP_STEP)

        if abs(updated) < 1e-9:
            updated = 0.0
            if current != 0.0:
                reverse_allowed_at = now + self.REVERSAL_NEUTRAL_S

        if current == 0.0 and target != 0.0 and now < reverse_allowed_at:
            updated = 0.0

        return updated, reverse_allowed_at

    def step_toward(self, left: float, right: float) -> None:
        """Take one acceleration step toward the requested signed commands."""
        now = time.monotonic()
        self._left, self._left_reverse_allowed_at = self._step_side(
            self._left, left, self._left_reverse_allowed_at, now
        )
        self._right, self._right_reverse_allowed_at = self._step_side(
            self._right, right, self._right_reverse_allowed_at, now
        )
        self._write_sides()

    def neutral(self) -> None:
        self._left = self._right = 0.0
        self._write_sides()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.neutral()
            time.sleep(0.1)
        finally:
            self.pca.deinit()


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Run this program from an interactive terminal or SSH session.")

    robot = FourWheelDrive()
    atexit.register(robot.close)

    def emergency_stop(_signum: int, _frame: object) -> None:
        robot.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)

    old_terminal_settings = termios.tcgetattr(sys.stdin)
    target_left = 0.0
    target_right = 0.0
    last_drive_key_time = 0.0

    print("Ready: w=forward, s=reverse, a=spin left, d=spin right.")
    print("Release to auto-neutral; Space/x=neutral; q=neutral and quit.")

    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            readable, _, _ = select.select(
                [sys.stdin], [], [], robot.CONTROL_INTERVAL_S
            )
            if readable:
                key = sys.stdin.read(1).lower()
                if key == "w":
                    target_left, target_right = 1.0, 1.0
                    last_drive_key_time = time.monotonic()
                elif key == "s":
                    target_left, target_right = -1.0, -1.0
                    last_drive_key_time = time.monotonic()
                elif key == "a":
                    target_left, target_right = -1.0, 1.0
                    last_drive_key_time = time.monotonic()
                elif key == "d":
                    target_left, target_right = 1.0, -1.0
                    last_drive_key_time = time.monotonic()
                elif key in {"x", " "}:
                    target_left = target_right = 0.0
                    robot.neutral()
                elif key == "q" or key == "\x04":  # q or terminal EOF
                    robot.neutral()
                    break

            driving = target_left != 0.0 or target_right != 0.0
            input_timed_out = (
                time.monotonic() - last_drive_key_time
                > robot.DRIVE_WATCHDOG_S
            )
            if driving and input_timed_out:
                target_left = target_right = 0.0
                robot.neutral()
            else:
                robot.step_toward(target_left, target_right)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)


if __name__ == "__main__":
    main()
