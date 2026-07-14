#!/usr/bin/env python3
"""Raised-wheel test for one THOR-2826 and Maxynos 45 A ESC.

The program controls PCA9685 channel 0. It keeps the bidirectional ESC at its
1500 us neutral, commands only forward during this basic test, and limits the
signal to 25%. Motor power never comes from the Raspberry Pi or laptop.
"""

from __future__ import annotations

import atexit
import argparse
import select
import signal
import sys
import termios
import time
import tty

ESC_CHANNEL = 0
PCA9685_ADDRESS = 0x40
PWM_FREQUENCY_HZ = 50

# Maxynos bidirectional normal-PWM mapping: 1000 us reverse, 1500 us neutral,
# and 2000 us forward. This test intentionally uses only neutral and forward.
NEUTRAL_PULSE_US = 1500
FULL_FORWARD_PULSE_US = 2000
ARM_TIME_S = 5.0

# The highest test selection is 25% forward (1625 us).
TEST_LEVELS = {
    "1": 0.10,
    "2": 0.15,
    "3": 0.20,
    "4": 0.25,
}

# Holding a number relies on normal keyboard repeat. If input disappears, the
# ESC receives a stop pulse within this time.
WATCHDOG_S = 0.80
LOOP_DELAY_S = 0.05


def throttle_to_pulse_us(throttle: float) -> int:
    """Pure calculation shared by hardware, simulation, and self-test modes."""
    throttle = max(0.0, min(0.25, throttle))
    return round(
        NEUTRAL_PULSE_US
        + throttle * (FULL_FORWARD_PULSE_US - NEUTRAL_PULSE_US)
    )


def pulse_to_duty_cycle(pulse_us: int) -> int:
    period_us = 1_000_000 / PWM_FREQUENCY_HZ
    return round((pulse_us / period_us) * 0xFFFF)


class HardwareOneMotorTester:
    def __init__(self) -> None:
        # Imported only in hardware mode, allowing --simulate and --self-test
        # to run on a laptop without Raspberry Pi libraries installed.
        import board
        import busio
        from adafruit_pca9685 import PCA9685

        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=PCA9685_ADDRESS)
        self.pca.frequency = PWM_FREQUENCY_HZ
        self._closed = False
        self.stop()

    def set_throttle(self, throttle: float) -> None:
        pulse_us = throttle_to_pulse_us(throttle)
        self.pca.channels[ESC_CHANNEL].duty_cycle = pulse_to_duty_cycle(pulse_us)

    def stop(self) -> None:
        self.set_throttle(0.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
            time.sleep(0.1)
        finally:
            self.pca.deinit()


class SimulatedOneMotorTester:
    """Print commands instead of accessing Raspberry Pi hardware."""

    def __init__(self) -> None:
        self._last_pulse_us: int | None = None
        self.stop()

    def set_throttle(self, throttle: float) -> None:
        pulse_us = throttle_to_pulse_us(throttle)
        if pulse_us == self._last_pulse_us:
            return
        self._last_pulse_us = pulse_us
        print(
            f"SIMULATION: channel={ESC_CHANNEL}, throttle={throttle:.0%}, "
            f"pulse={pulse_us} us, duty={pulse_to_duty_cycle(pulse_us)}"
        )

    def stop(self) -> None:
        self.set_throttle(0.0)

    def close(self) -> None:
        self.stop()


def run_self_test() -> None:
    """Verify every test level produces the expected command."""
    expected_pulses = {
        0.00: 1500,
        0.10: 1550,
        0.15: 1575,
        0.20: 1600,
        0.25: 1625,
    }
    for throttle, expected in expected_pulses.items():
        actual = throttle_to_pulse_us(throttle)
        assert actual == expected, (throttle, actual, expected)
        duty = pulse_to_duty_cycle(actual)
        assert 0 <= duty <= 0xFFFF
        print(f"PASS: {throttle:.0%} -> {actual} us -> duty {duty}")
    print("All motor-command calculation checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="print commands without accessing Raspberry Pi hardware",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check all pulse calculations and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not sys.stdin.isatty():
        raise SystemExit("Run this from an interactive terminal or SSH session.")

    tester = SimulatedOneMotorTester() if args.simulate else HardwareOneMotorTester()
    atexit.register(tester.close)

    def emergency_stop(_signum: int, _frame: object) -> None:
        tester.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)

    if args.simulate:
        print("SIMULATION MODE: no Raspberry Pi hardware will be accessed.")
    else:
        print(
            f"Sending a {NEUTRAL_PULSE_US} us neutral signal "
            f"on channel {ESC_CHANNEL}."
        )
        print("All wheels must be raised. Apply ESC power only when ready.")
        print(f"Waiting {ARM_TIME_S:.0f} seconds for ESC arming...")
        time.sleep(ARM_TIME_S)
    print("Hold 1=10%, 2=15%, 3=20%, or 4=25% throttle.")
    print("Release the key to auto-stop; Space/x=stop; q=stop and quit.")

    old_terminal_settings = termios.tcgetattr(sys.stdin)
    active_throttle = 0.0
    last_drive_key_time = 0.0

    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], LOOP_DELAY_S)
            if readable:
                key = sys.stdin.read(1).lower()
                if key in TEST_LEVELS:
                    active_throttle = TEST_LEVELS[key]
                    last_drive_key_time = time.monotonic()
                    tester.set_throttle(active_throttle)
                elif key in {" ", "x", "s"}:
                    active_throttle = 0.0
                    tester.stop()
                elif key == "q" or key == "\x04":
                    tester.stop()
                    break

            timed_out = time.monotonic() - last_drive_key_time > WATCHDOG_S
            if active_throttle > 0.0 and timed_out:
                active_throttle = 0.0
                tester.stop()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)


if __name__ == "__main__":
    main()
