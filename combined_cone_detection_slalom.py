#!/usr/bin/env python3
"""Combined cone detection and autonomous slalom driving for Raspberry Pi OS.

Run this file on the Raspberry Pi after calibrating the camera.  The complete
camera, navigation, motor-control, and safety implementation lives in
``autonomous_cone_slalom.py`` so both filenames always run the same program.
"""

from autonomous_cone_slalom import main


if __name__ == "__main__":
    main()
