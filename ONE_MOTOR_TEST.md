# One-motor raised-wheel test

This test uses one THOR-2826 motor, one MAXYNOS 45 A bidirectional ESC,
PCA9685 channel 0, and the Raspberry Pi 5. The basic test commands neutral and
low forward power only; use the four-wheel controller after this succeeds.

## Safe PWM values

For this ESC, 1000 microseconds is full reverse, not stop. The commands are:

| Command | Pulse width |
| --- | ---: |
| Full reverse | 1000 microseconds |
| Neutral / brake | 1500 microseconds |
| Full forward | 2000 microseconds |

The test program starts at 1500 microseconds and never exceeds 25% forward.

## Battery-free checks

Run the calculation check on any computer with Python 3:

```bash
python3 one_motor_test.py --self-test
```

It verifies neutral and the four forward test levels: 1500, 1550, 1575, 1600,
and 1625 microseconds.

For an interactive simulation without Raspberry Pi hardware:

```bash
python3 one_motor_test.py --simulate
```

## Wiring with all power disconnected

| Raspberry Pi 5 | PCA9685 |
| --- | --- |
| Pin 1, 3.3 V | VCC |
| Pin 3, GPIO 2/SDA | SDA |
| Pin 5, GPIO 3/SCL | SCL |
| Pin 6, GND | GND |

Do not put motor voltage on the PCA9685 V+ terminal.

Connect the ESC control lead as follows:

- White ESC wire -> PCA9685 channel 0 signal
- Black ESC wire -> PCA9685 channel 0 ground

The ESC has no BEC. Keep the Pi powered separately over USB-C, but keep the
Pi, PCA9685, and ESC signal grounds common.

Connect the ESC's three motor outputs to the THOR motor using the supplied
4 mm-to-3.5 mm adapters. Keep middle connected to middle. With power removed,
swap the two outer phases if the raised wheel's direction is wrong.

## Motor power

Use only the already verified, protected motor source that remains within the
THOR-2826's published 7-15 V range. Never connect ESC motor-power wires to the
Pi, PCA9685, USB, or a computer.

## Run the hardware test

With ESC motor power disconnected:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python one_motor_test.py
```

The program immediately sends neutral. After it announces the neutral arming
period, apply motor power with the wheel securely raised.

- Hold `1`, `2`, `3`, or `4` for 10%, 15%, 20%, or 25% forward.
- Release the key to command neutral automatically.
- Space or `x` commands neutral.
- `q` commands neutral and exits.

Use a physical motor-power disconnect if the motor moves at startup, an ESC
alarms continuously, wiring heats, or anything smells wrong. Disconnect motor
power before touching wiring.
