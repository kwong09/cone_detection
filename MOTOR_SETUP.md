# Raspberry Pi four-wheel bidirectional motor setup

This setup uses four MAXYNOS THOR-2826 1000 kV, 19:1 motors, four MAXYNOS
45 A bidirectional ESCs, one Raspberry Pi 5, and one PCA9685 PWM board. Each
motor requires its own ESC. All four ESCs share one correctly sized and
protected motor-power source through a fused distribution system.

## System layout

```text
Motor power source (verified 7-15 V for the THOR-2826)
  -> main fuse
  -> physical emergency disconnect
  -> positive and negative power distribution
       -> protected branch 1 -> ESC 1 -> front-left motor
       -> protected branch 2 -> ESC 2 -> rear-left motor
       -> protected branch 3 -> ESC 3 -> front-right motor
       -> protected branch 4 -> ESC 4 -> rear-right motor

Separate regulated Raspberry Pi supply
  -> Raspberry Pi 5 -> I2C -> PCA9685
                              -> channel 0 -> ESC 1 signal
                              -> channel 1 -> ESC 2 signal
                              -> channel 2 -> ESC 3 signal
                              -> channel 3 -> ESC 4 signal
```

Do not connect four motors to one ESC. Do not daisy-chain ESC power through
another ESC. Do not use an unfused lightweight one-to-four splitter for the
motor current.

## Motor-power distribution

The THOR-2826 1000 kV motor is rated at 21 A continuous, so four motors can
draw about 84 A total at their rated operating points and more briefly during
startup or a stall. The battery or supply, connector, main fuse, emergency
disconnect, distribution hardware, wiring, and return path must all be sized
as one system.

With all power disconnected:

1. Connect motor-source positive to a main fuse.
2. Connect the fuse output to a physical emergency disconnect.
3. Connect the disconnect output to a positive distribution bus.
4. Feed each ESC's thick positive power wire from its own protected branch.
5. Connect all four ESC thick negative power wires to a negative distribution
   bus, then connect that bus to motor-source negative.

Have a qualified mentor select the main and branch fuse ratings from the
actual battery, wire gauge, connector ratings, expected current, and fuse
time-current curves. A fuse must protect the wire and downstream equipment;
do not choose it merely because the ESC says 45 A.

The ESC has an XT60 power connector. If the motor source uses a different
connector, use a correctly rated adapter or distribution assembly. Never cut
or modify a Smart battery's data contact, and never reverse power polarity.

The THOR motors are specified for 7-15 V. Only use a motor supply already
verified to remain in that range under every charge condition. A nominal
14.8 V 4S LiPo reaches 16.8 V fully charged and is not within that published
motor range unless MAXYNOS has approved that exact use in writing.

## Motor connections

Each ESC controls exactly one motor through three phase wires. Use the three
included 4 mm-to-3.5 mm bullet adapters.

- ESC 1 -> front-left motor
- ESC 2 -> rear-left motor
- ESC 3 -> front-right motor
- ESC 4 -> rear-right motor

Keep the middle motor phase connected to the middle ESC phase. If a raised
wheel rotates opposite the robot's desired direction, disconnect all motor
power and swap only the two outer phase wires for that motor. Never change a
motor connection while the system is powered.

## Raspberry Pi to PCA9685

| Raspberry Pi 5 | PCA9685 |
| --- | --- |
| Pin 1, 3.3 V | VCC |
| Pin 3, GPIO 2/SDA | SDA |
| Pin 5, GPIO 3/SCL | SCL |
| Pin 6, GND | GND |

Enable I2C in Raspberry Pi Configuration. Do not connect motor voltage to the
Pi or to the PCA9685 V+ terminal. Power the Pi from its own regulated USB-C
supply.

## Four ESC control connections

The MAXYNOS 45 A ESC has no BEC. Its thin control wires are:

- White: PWM signal
- Black: signal ground

| Wheel | PCA9685 channel | ESC white wire | ESC black wire |
| --- | ---: | --- | --- |
| Front left | 0 | Channel 0 signal | Channel 0 ground |
| Rear left | 1 | Channel 1 signal | Channel 1 ground |
| Front right | 2 | Channel 2 signal | Channel 2 ground |
| Rear right | 3 | Channel 3 signal | Channel 3 ground |

All four ESC signal grounds, PCA9685 ground, and Raspberry Pi ground must be
common. The thin black signal wire is required even though each ESC also has a
thick negative motor-power wire.

## PWM commands

The ESC uses standard 50 Hz RC PWM:

| Command | Pulse width |
| --- | ---: |
| Full reverse | 1000 microseconds |
| Neutral / brake | 1500 microseconds |
| Full forward | 2000 microseconds |

The program starts every ESC at 1500 microseconds, limits the first test to
20%, ramps commands, pauses at neutral before reversing, and commands neutral
if keyboard input disappears.

## Install and run

With motor power disconnected:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python motor_control.py
```

Start the program first so all four ESCs receive neutral. Only then apply
motor power.

Controls:

| Key | Left motors | Right motors | Motion |
| --- | --- | --- | --- |
| `w` | Forward | Forward | Forward |
| `s` | Reverse | Reverse | Reverse |
| `a` | Reverse | Forward | Spin left |
| `d` | Forward | Reverse | Spin right |
| Space or `x` | Neutral | Neutral | Stop |
| `q` | Neutral | Neutral | Stop and exit |

## First four-motor test

1. Disconnect motor power and inspect every power and signal connection.
2. Verify polarity, fuse installation, emergency disconnect, and the measured
   motor-source voltage.
3. Put the chassis on a strong stand with all four wheels clear of the floor.
4. Keep hands, hair, clothing, tools, and wires away from every wheel.
5. Start `motor_control.py`; confirm that it announces a 1500-microsecond
   neutral arming period.
6. Apply motor power. Immediately use the physical disconnect if any wheel
   moves, an ESC continuously alarms, wiring heats, or anything smells wrong.
7. Briefly hold `w`. Confirm that all wheels move the robot-forward direction.
8. If one wheel is backward, disconnect motor power and swap that motor's two
   outer phase wires.
9. Test `s` briefly, then `a` and `d`. The first-test ceiling is 20%.
10. Measure current and check motor, ESC, connector, and wiring temperatures
    before increasing the software limit.

The physical emergency disconnect is the safety stop. A keyboard command,
SSH connection, watchdog, or software process is not a substitute.
