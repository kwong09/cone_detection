# Current Parts Inventory

Last updated: **July 13, 2026**

These are the parts the team currently has available for the robot.

| Date added | Part | Source / description | Unit or package price | Quantity on hand | Cost |
| --- | --- | --- | ---: | ---: | ---: |
| 7/13/2026 | Motors | [Maxynos THOR 2826 planetary gear brushless motor, 1000 KV, 19:1](https://www.amazon.com/dp/B0FL7X27B4) | $57.99 | 4 motors | $231.96 |
| 7/13/2026 | ESCs | [30 A brushless ESC speed controllers, two per package](https://www.amazon.com/dp/B0CZNM4H7H) | $18.59/package | 2 packages (4 ESCs) | $37.18 |
| 7/13/2026 | Bidirectional ESCs | [MAXYNOS 45 A bidirectional brushless ESC, 2–6S, XT60](https://www.amazon.com/dp/B0FPWNBR88) | $23.99 | 4 ESCs | $95.96 |
| 7/13/2026 | Wheel rubber | [TORRAMI neoprene rubber strip, 1/8 in thick × 2 in wide × 10 ft](https://www.amazon.com/dp/B0859FZR2S) | $14.38 | 1 roll | $14.38 |
| 7/13/2026 | Batteries | Other team | $40.00 | 2 batteries | $80.00 |
| 7/13/2026 | Charger | Other team | $60.00 | 1 charger | $60.00 |
| 7/13/2026 | Screws | [smseace screw and nut assortment](https://www.amazon.com/dp/B0DN1676NW) | $7.99 | 1 package | $7.99 |
| 7/13/2026 | Pillow bearings | [25 mm bore mounted pillow block bearings, two per package](https://www.amazon.com/dp/B0G4D36Z1X) | $6.69/package | 1 package (2 bearings) | $6.69 |
| 7/13/2026 | Front rotation shaft | [McMaster-Carr 4138N85](https://www.mcmaster.com/products/4138n85/) | $15.64 | 1 shaft | $15.64 |
| 7/13/2026 | Front rotation collars | [McMaster-Carr 6056N18](https://www.mcmaster.com/products/6056n18/) | $2.00 | 2 collars | $4.00 |
| 7/13/2026 | 1/8 in rivets | [McMaster-Carr 97447A020](https://www.mcmaster.com/products/97447a020/) | $10.57 | 1 package | $10.57 |
|  |  | **Total recorded cost** |  |  | **$564.37** |

## Required electrical items not yet recorded as on hand

Only parts needed to connect the listed battery to the four selected MAXYNOS
ESCs are included. The Raspberry Pi 5 and PCA9685 are treated as already on hand
because the project setup identifies them as the existing controller.

| Required item | Bare-minimum specification | Quantity needed | Selected part / status |
| --- | --- | ---: | --- |
| Main circuit breaker/disconnect | Manual-reset, switchable, 100 A, at least 32 V DC | 1 | [Blue Sea 7144, 100 A](https://www.amazon.com/dp/B000KOTALG) |
| Battery mating lead | Genuine IC3 Device connector with copper wire; Spektrum SPMXCA305 or exact equivalent | 1 | [Amazon search: SPMXCA305](https://www.amazon.com/s?k=SPMXCA305) |
| Fused power-distribution block | Common positive and negative buses, cover, at least 100 A total, 30 A per circuit and 32 V DC | 1 | [Blue Sea 5025](https://www.amazon.com/dp/B000THQ0CQ) |
| Motor-branch fuses | 30 A ATO/ATC automotive blade fuses | 4 | [Amazon search: 30 A ATO/ATC fuses](https://www.amazon.com/s?k=30A+ATO+ATC+automotive+blade+fuse) |
| ESC mating leads | Genuine XT60 mating pigtails with 12 AWG copper wire | 4 | [Amazon search: 12 AWG XT60 pigtails](https://www.amazon.com/s?k=12+AWG+XT60+pigtail) |
| Power wire and terminals | 8 AWG pure-copper red/black main wire; 12 AWG pure-copper red/black branch wire; matching crimped ring terminals and heat shrink | As measured | [8 AWG wire](https://www.amazon.com/s?k=8+AWG+pure+copper+red+black+wire) and [12 AWG wire](https://www.amazon.com/s?k=12+AWG+pure+copper+red+black+wire) |
| Raspberry Pi power bank and USB-C cable | Regulated USB-C output of 5 V at 3 A continuously | 1 | [Amazon search: USB-C 5 V 3 A power bank](https://www.amazon.com/s?k=USB-C+power+bank+5V+3A+output) |
| Control jumper wires | 22–24 AWG connections for Pi-to-PCA9685 I2C and PCA9685-to-four-ESC signal/ground | As needed | [Amazon search: female-female Dupont jumper wires](https://www.amazon.com/s?k=female+female+Dupont+jumper+wires) |

## Inventory notes

- “Quantity on hand” records physical items when the package size is known; otherwise it records packages or rolls.
- The ESC listing contains two ESCs per package, so two purchased packages provide four ESCs.
- Each MAXYNOS 45 A ESC controls one motor; four ESCs provide bidirectional control for all four motors. The recorded unit price is the MAXYNOS direct-store price on July 13, 2026; verify the Amazon checkout price separately.
- Each selected MAXYNOS ESC includes the three bullet adapters needed for one
  THOR motor, so separate motor bullet adapters are not required.
- The pillow-bearing listing contains two bearings per package.
- Battery and charger model details were not included in the original list. Add them here when confirmed.
- Prices are the recorded prices from July 13, 2026 and may not include tax or shipping.
