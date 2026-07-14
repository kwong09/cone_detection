# Webcam Cone Detection

This prototype detects the team's **saturated red traffic cones** using their
color and tapered shape. Its color range is tuned for the current test-room
lighting and rejects the orange tones in skin and wood. It
does not need an internet connection or a trained AI model.

## Saved iterations

- `cone_detection_iteration_1.py`: reliably detects the red cones and shows the
  camera and mask side by side. Its pass/slalom estimate is not reliable enough
  for driving because it treats disappearance of a nearby cone as a pass without
  confirming the cone moved past the correct side of the robot.
- `cone_detection_iteration_2.py`: keeps the same detector and adds head-on
  slalom feedback. When the robot moves right, the cone must move toward the
  left of the image; when the robot moves left, the cone must move toward the
  right. A pass is counted only after the close cone reaches that expected side
  and leaves the view.
- `cone_detection_iteration_3.py`: adds calibrated distance estimation. It
  approaches a cone head-on, starts the alternating turn at a measured range,
  increases the turn urgency when close, and confirms the cone passed on the
  correct side.
- `cone_detection_iteration_4.py`: accounts for the camera rotating with the
  robot. It measures cone motion relative to the start of the turn, recognizes
  the cone moving away after the closest point, immediately countersteers in
  the other direction, and temporarily ignores the passed cone. This is the
  final webcam-focused version.
- `cone_detection_iteration_5_pi.py`: uses Picamera2 for a CSI Arducam on
  Raspberry Pi 5, falls back to OpenCV for a USB Arducam, and supports both a
  desktop dashboard and headless feedback. This is the current version.
- `cone_detector.py`: convenience launcher for the newest iteration (currently
  Iteration 5).
- `autonomous_cone_slalom.py`: combines Iteration 5 detection/navigation with
  the four-ESC controller and commands a low-speed, forward-only slalom.
- `combined_cone_detection_slalom.py`: clearly named Raspberry Pi launcher for
  the same combined cone-detection and autonomous-slalom program.

## 1. Install

Open a terminal in this project folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On macOS, allow your terminal application to use the camera if prompted. Camera
permission can also be enabled in **System Settings → Privacy & Security → Camera**.

## 2. Run

```bash
python3 cone_detector.py
```

To compare with the preserved first version, run:

```bash
python3 cone_detection_iteration_1.py
```

The dashboard shows the annotated live camera on the left and the red-only mask
on the right. Hold a red cone in view. A green box and the words `CONE LEFT`,
`CONE CENTER`, or `CONE RIGHT` should appear.

Iteration 2 starts by steering right around the first head-on cone. It expects
that cone to move left in the camera view. The banner becomes more urgent if a
close cone is still centered. After the cone reaches the expected side and
leaves the view, the program switches left for the next cone and reverses the
expected image motion. It continues alternating for the slalom.

## Iteration 3 distance calibration

Distance requires one calibration for each camera, resolution, and field of
view. Measure the cone's full physical height. The default is 30.5 cm (12 in),
so pass the real height if yours differs. For example, for a 22.9 cm cone:

```bash
python3 cone_detector.py --cone-height-cm 22.9
```

1. Measure exactly 100 cm (39.4 in) from the camera lens.
2. Put one upright cone at that mark with its full top and base visible.
3. Start the program with the correct `--cone-height-cm` value.
4. Press **C** while the detector has one clean box around the cone.

The program saves `cone_camera_calibration.json` and loads it automatically on
future runs. Pressing **C** again replaces the calibration. Recalibrate after
changing the camera, resolution, lens, camera mode, or digital crop.

The default slalom timing is:

- Begin the alternating turn at 130 cm.
- Demand a hard turn at 80 cm if the cone has not moved to the passing side.
- Arm pass detection at 60 cm once the cone is on the correct side.

These values must be tuned at low speed for the final robot. Example:

```bash
python3 cone_detector.py --turn-start-cm 150 --hard-turn-cm 90 --pass-distance-cm 65
```

## Iteration 4 turning-camera behavior

The camera turns with the robot, so cone location is not compared only with a
fixed left or right image zone. Iteration 4 records the cone's horizontal
position when the turn begins. For a right turn it expects the cone to slide
left relative to that recorded position; a left turn expects the opposite.

After the cleared cone reaches its closest point and begins getting smaller or
farther away, the program counts the pass and immediately commands the opposite
turn. It ignores detections for 12 frames while the camera rotates toward the
next cone. This can be adjusted if the countersteer is too short or long:

```bash
python3 cone_detector.py --countersteer-frames 18
```

## Raspberry Pi 5 and Arducam

Iteration 5 first tries Picamera2, which is the normal backend for an Arducam
connected to a Pi 5 CAM/DISP ribbon connector. It falls back to OpenCV for a USB
Arducam.

Install the Raspberry Pi OS packages:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-venv
```

Create the environment so it can see Raspberry Pi OS's camera libraries:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Check that Raspberry Pi OS detects the Arducam before running the program:

```bash
rpicam-hello --list-cameras
rpicam-hello --timeout 0
```

For a ribbon-connected CSI camera, run:

```bash
python3 cone_detector.py --backend picamera2
```

For a USB Arducam, run:

```bash
python3 cone_detector.py --backend opencv
```

The calibration from the laptop is not valid for the Arducam. Measure the cone,
place it 100 cm from the Arducam lens, and recalibrate using a 30-frame median:

```bash
python3 cone_detector.py --backend picamera2 --cone-height-cm 30.5 --calibrate
```

For Raspberry Pi OS Lite or operation without a monitor, calibrate first and
then run:

```bash
python3 cone_detector.py --backend picamera2 --headless
```

Use `--vflip` if the installed camera is upside down. Use `--camera 1` if
`rpicam-hello --list-cameras` reports the desired sensor at index 1.

## Autonomous motor control

First calibrate the Arducam with motor power disconnected:

```bash
python3 cone_detector.py --backend picamera2 --cone-height-cm 30.5 --calibrate \
  --calibration-file pi_cone_camera_calibration.json
```

For the first motor test, raise all four wheels, keep the physical kill switch
in reach, and verify that `RIGHT` turns the chassis right and `LEFT` turns it
left. The combined program uses the supplied controller's **1400 us stop** and
**1460 us starting** pulses. Do not run it if the installed ESCs instead use
1500 us neutral; update and re-test the pulse calibration first. The `--drive`
flag is deliberately required:

```bash
python3 combined_cone_detection_slalom.py --backend picamera2 --drive
```

The autonomous program opens a live annotated camera-and-mask dashboard and
starts **paused**. Press **G** to begin driving, **Space** or **S** to stop and
pause, **R** to reset the slalom while stopped, and **Q** or **Escape** to stop
and quit. Losing camera frames, a normal remote-terminal disconnect, or
pressing Ctrl+C also commands the motor stop pulse. Use `--headless` only when
no live window is needed; headless autonomy starts immediately.

The autonomous program deliberately uses a separate Pi calibration file so a
calibration committed from another camera cannot start the robot. The initial
settings cap requested throttle at 18%, ramp motor pulses, stop immediately on
camera loss, stop after two seconds of searching without seeing the next cone,
and stop on Ctrl+C. Once direction and stopping have been verified with raised
wheels, test on the ground at low speed with wide cone spacing. If the chassis
turns opposite the printed direction, swap `RIGHT_TURN_MOTORS` and
`LEFT_TURN_MOTORS` in the autonomous script before continuing.

In the detector-only dashboard, press **R** to reset the sequence and **Q** or
**Escape** to stop.

If the wrong camera opens, try:

```bash
python3 cone_detector.py --camera 1
```

The mask is always displayed beside the camera; no extra option is needed.
To make the combined window narrower, use `--display-width 1200`.

The detector has a second close-range shape path for cones whose base or side
is partly outside the image. It keeps the stricter tapered-shape requirements
for small distant objects, while allowing a large cone low in the image to
remain outlined as the robot approaches it.

If distant cones are ignored, reduce the minimum area, for example:

```bash
python3 cone_detector.py --min-area 150
```

Lower values see smaller cones but also increase false detections.

## Lighting tips

- Test in lighting similar to the competition area.
- The current narrow color range is intentionally specific to these red cones
  and this room. Retune it when the camera or lighting changes substantially.
- Keep auto-exposure from pointing directly toward a bright window.
- In the mask panel, the red cone should appear mostly white and people, the red
  ball, and the background should remain black.

## Raspberry Pi / Arducam path

The detection logic is separated into `detect_cones(frame)`. When moving to the
Pi 5, only the code that supplies each `frame` needs to change. A CSI Arducam
will normally use Raspberry Pi's Picamera2/libcamera stack rather than
`cv2.VideoCapture`. Install OpenCV from Raspberry Pi OS (`python3-opencv`) and
use the camera's supported Picamera2 capture method; keep the detector and
drawing functions unchanged.

Color detection is a good first milestone, but it can mistake other red
objects for cones. Once webcam footage has been collected from the actual
course, a small trained object-detection model can replace this first detector
while preserving the same camera-to-detection structure.
