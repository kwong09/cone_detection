# Power plan for the Spektrum SPMX224S50 battery

## Battery you already have

- Spektrum SPMX224S50 Smart G2
- 4-cell (4S) lithium-polymer battery
- 14.8 V nominal, 16.8 V when fully charged
- 2200 mAh / 32.56 Wh
- 50C advertised continuous discharge rating
- IC3 connector and 12 AWG leads
- Requires a Spektrum Smart charger with G2 support

This is a LiPo battery, not a lithium-iron (LiFePO4) battery. The chemistry must
be selected correctly whenever it is charged.

## Why it should not connect directly

The THOR-2826 datasheet specifies a 7-15 V operating range. A fully charged 4S
LiPo is 16.8 V. The 14.8 V printed on the battery is its average voltage, not
its highest voltage.

The software throttle limit does not change the voltage coming out of the
battery, and a plug adapter only changes the connector shape. Neither protects
the motors from a supply above their published maximum.

Before buying power hardware, ask Maxynos in writing whether this exact
THOR-2826 1000 kV motor may be used directly with a fully charged 4S LiPo
(16.8 V). If Maxynos confirms that use, save their response and the large
motor-voltage reducer below may not be necessary.

## Add-ons if Maxynos does not approve direct 4S use

The battery can be retained by adding this power chain:

```text
4S battery
  -> IC3-compatible connection
  -> main fuse and physical disconnect
  -> high-power 12 V DC-DC step-down converter
  -> fused four-way power distribution
  -> four ESCs
  -> four motors
```

A DC-DC step-down converter is an electronic pressure reducer: it accepts the
battery's changing 16.8-to-lower voltage and provides about 12 V to the ESCs.
For this robot it must be an industrial/automotive-size unit designed for motor
loads, approximately 12 V output and at least 100 A continuous output, with
startup-current margin and adequate cooling. That is roughly a 1.2 kW device.
A small circuit-board regulator advertised for a few amps is not suitable.

The four motors can draw about 84 A total at their rated operating points. The
converter, connector, distribution, wire, fuse, and switch ratings must be
chosen together so the weakest part is protected. Do not use an unfused cheap
one-to-four splitter.

## Connectors and distribution

The battery has IC3 while the four current ESCs have XT60 inputs. Use a proper
IC3/EC3-compatible device lead into the protected converter/distribution
assembly. Do not cut or replace the Smart battery's connector. The Smart data
contact is used for charging/telemetry and is normally unused by a basic power
adapter.

The converter output then needs four separately protected XT60 branches, one
for each ESC. Have a qualified mentor select the connector, wire gauge, branch
protection, main protection, and disconnect based on the chosen converter's
manual.

## Raspberry Pi power

The Raspberry Pi must not receive 12-16.8 V. The simplest early setup is to
power the Pi separately with a suitable USB-C power bank or official-style Pi 5
USB-C supply. A mobile final robot can instead use a separate 5.1 V, 5 A DC-DC
USB-C supply designed for Raspberry Pi 5. Do not power the Pi from the four ESC
red BEC wires and do not connect those red wires together.

## Expected runtime

The pack stores 32.56 Wh. Four motors at their published rated input could use
about 1260 W total, which would empty this small pack in roughly 1.5 minutes in
an ideal full-load calculation. Real driving at lower average power may last
longer, but this pack should be treated as a short-test battery until current
and runtime are measured.

## Charging

Smart G2 batteries have no separate balance connector and require a compatible
Spektrum Smart charger with current G2 firmware. Charge it as a 4S LiPo only,
on a nonflammable surface and according to Spektrum's safety instructions.
The charger's storage mode is for battery care; it is not a dependable motor
voltage regulator.
