# Off-Road Autonomous Challenge Requirements

## Challenge Summary

Design and build an autonomous ground vehicle that completes a rough off-road course. During the run, the vehicle collects tennis balls, scores them into a bucket, and finishes by pulling itself up on a horizontal bar.

## Course Sections

The run is one continuous course in this order:

1. **Slalom**
   - Vehicle weaves through cones or gates without contact.
   - Balls may be collected before the slalom and carried through the course for bonus points.

2. **Washboard**
   - Fixed transverse bumps of varying height.
   - Tests suspension, stability, and body control.
   - IMU pitch and roll are used for stability scoring.

3. **Hill**
   - Vehicle climbs an incline and descends the other side.
   - Leaving the hill surface, except by driving straight off the far side, fails the run.

4. **Ball Task**
   - Tennis balls are staged on the course.
   - Vehicle collects balls and shoots them into a bucket from one of three scoring lines.

5. **Final Climb**
   - Vehicle engages a horizontal bar at the finish and pulls itself up.

## Driving Section Scoring

Scores are awarded at the highest tier reached.

| Tier | Description | Hill / Washboard | Slalom |
| --- | --- | ---: | ---: |
| T3 | Cleared fully, autonomously | 15 | 10 |
| T2 | Autonomous but partial, such as hill up but not down or cone knocked | 10 | 7 |
| T1 | Completed non-autonomously | 5 | 3 |
| T0 | Not cleared fully and cleanly, off-track, cone hit, etc. | 0 | 0 |

## Ball Task Scoring

Points are awarded per ball landed in the bucket.

| Line | Points Per Ball | Points If Carried Through Course |
| --- | ---: | ---: |
| Line 1, closest | 3 | 4.5 |
| Line 2, middle | 5 | 7.5 |
| Line 3, farthest | 12 | 18 |

Additional rules:

- Autonomous collection and shot: full points.
- Manual assist: one-third points.
- Missed bucket: 0 points.

## Climb Scoring

| Tier | Result | Points |
| --- | --- | ---: |
| T3 | Fully raised | 20 |
| T2 | Off the ground | 10 |
| T1 | Hooked, but no lift | 6 |
| T0 | No hook | 0 |

Additional rules:

- Autonomous climb: full points.
- Assisted climb: half points.

## Washboard Stability Scoring

The IMU must measure pitch and roll during the run.

Tilt is calculated at each time step:

```text
tilt = sqrt(roll^2 + pitch^2)
```

Then the squared tilt values are averaged:

```text
avg_of_squares = (tilt_1^2 + tilt_2^2 + tilt_3^2 + ... + tilt_N^2) / N
```

The preliminary tilt score is:

```text
preliminary_tilt_score = sqrt(avg_of_squares)
```

The final stability score is:

```text
final_stability_score = -1 * preliminary_tilt_score * 2.5
```

This final stability score is subtracted from the total points. The IMU may not have its own gyro or suspension.

## Speed Bonus

| Finish Time | Bonus |
| --- | ---: |
| Under 1:00 | +50 |
| Under 1:45 | +15 |
| No finish or over 1:45 | 0 |

## Run Format

- Each competition has 3 attempts.
- Best score counts.
- If the robot does not finish, it still scores for completed sections.
- After "go," no contact with the robot is allowed.
- Any contact with the robot ends the run, and the score stands from that point.
- The run ends at completed final climb, time expiration, or judge's stop.
- No restarts.
- The run fails if the vehicle leaves the washboard or hill at all, except by driving straight off the far side.

## Safety Requirements

1. Approved safety mechanism, such as kill switch or tether, must be approved by judges.
2. Battery must be secured against ejection if the robot tips.
   - Use strap, bolt, bracket, or similar.
   - Battery system must include a fuse or breaker rated for the system.
   - Batteries and power systems must be approved by judges before purchase.
3. No exposed or dragging wiring.
   - Route and secure all leads.
4. A mentor or judge may stop any run for safety.
   - This is not a penalty.
   - The run may be restarted at their discretion.
5. Safety glasses are required in the build area at all times.

## Vehicle Size

- Footprint limit: **15 in x 15 in**, measured by contact patch centers of wheels in each corner.
- Maximum height at start: **8 in**.
- Arms, climbers, and other mechanisms may deploy after starting.
- Judges will clarify measurement for unusual designs.

## Vehicle Components

Teams must design and fabricate their own chassis and suspension.

COTS parts allowed:

- Motors
- Wheels
- Electronics

Parts teams are expected to design and fabricate, as determined by judges:

- Chassis
- Body parts
- Suspension arms
- Motor mounts
- Electronics enclosures
- Mechanical intakes
- Arms
- Grabbers
- Shooters
- Steering mechanicals

The majority of vehicle components should be custom designed and fabricated.

## Inspection Checklist

The robot must pass inspection before the first run.

Self-check:

- Fits inside vehicle size boundaries.
- Safety mechanism approved.
- Battery secured.
- Fuse or breaker installed and rated for the system.
- No exposed or loose wiring.
- IMU installed.
- Code reads pitch and roll during runs.

## Current Prototype Goal

By Wednesday, the immediate goal is not the full competition robot. The near-term goal is:

- Build a small chassis with wheels.
- Use pre-made parts where helpful.
- Get a basic manual drive-through working.
- Keep the robot well under the 15 in x 15 in size limit if possible.
- Use this first robot as a baseline drivetrain prototype.
