# Hardware Component Selection and GPIO Mapping

This document details the specific components chosen and the GPIO pin assignments for the RP2040 Pizza Detector project.

**Date:** 2025-04-25

## 1. Component Selection (MPNs)

| Reference | Component Type                  | Proposed MPN                     | Description                                                                 | Notes                                                               |
| :-------- | :------------------------------ | :------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------ |
| U3        | Buck-Boost Regulator            | `TPS63031DSKR`                   | TI, Adjustable Buck-Boost, 2.5A, SON-10                                     | Matches `mdis2.text`                                                |
| Q1        | P-Channel MOSFET                | `DMP2104L-7`                     | Diodes Inc., -20V, -3.6A, P-Channel, SOT-23                                 | Matches `mdis2.text`                                                |
| L1        | Buck-Boost Inductor             | `744043152` (Example)            | Würth Elektronik, 1.5µH, Shielded Power Inductor, ~2.5A Saturation, SMD   | Value matches `mdis2.text`. Verify current/DCR against TPS63031 datasheet. |
| C_in      | Buck-Boost Input Cap            | `GRM188R61A106KE69D` (Example)   | Murata, 10µF, 10V, X5R Ceramic, 0603                                       | Value matches `mdis2.text`.                                         |
| C_out     | Buck-Boost Output Cap           | `GRM21BR61A226ME15L` (Example)   | Murata, 22µF, 10V, X5R Ceramic, 0805                                       | Value matches `mdis2.text`.                                         |
| ZD1       | Zener Diode (Gate Protection)   | `BZT52C5V1S-7-F` (Example)       | Diodes Inc., 5.1V Zener, SOD-323                                            | Value matches `mdis2.text`.                                         |
| R_G       | MOSFET Gate Resistor            | `RC0603FR-0710KL` (Example)      | Yageo, 10kΩ, 1%, 0603                                                       | Value matches `mdis2.text`.                                         |
| R_S       | MOSFET Source Resistor          | `RC0603FR-07100KL` (Example)     | Yageo, 100kΩ, 1%, 0603                                                      | Value matches `mdis2.text`.                                         |
| U1        | RP2040 Module                   | `SC0915`                         | Raspberry Pi Pico (No WiFi/BT)                                              |                                                                     |
| CAM1      | OV2640 Camera Module            | Varies (e.g., `MH-ET LIVE`)      | Standard 2MP OV2640 module, 24-pin connector                                | Ensure connector pitch/pinout matches PCB footprint.                |
| C_decoup  | Decoupling Capacitors           | `CC0603KRX7R9BB104` (Example)    | Yageo, 100nF (0.1µF), 50V, X7R Ceramic, 0603                               | Place near IC power pins (RP2040, OV2640).                          |
| R_pullup  | I2C Pull-up Resistors           | `RC0603FR-074K7L` (Example)      | Yageo, 4.7kΩ, 1%, 0603                                                      | For I2C SDA and SCL lines.                                          |

*Note: Example MPNs should be verified for current availability and suitability (especially inductor and capacitors based on final calculations/simulations if performed).*

## 2. RP2040 (Raspberry Pi Pico) GPIO Assignment

| Signal (OV2640) | Pico Pin # | Pico GPIO # | Function        | Notes                                     |
| :-------------- | :--------- | :---------- | :-------------- | :---------------------------------------- |
| GND             | 3, 8, etc. | GND         | Ground          | Connect all GNDs                          |
| 3V3             | 36         | 3V3 (OUT)   | Power (Input)   | Power for Camera (from Buck-Boost output) |
| SDA             | 4          | GP2         | I2C Data        | I2C1                                      |
| SCL             | 5          | GP3         | I2C Clock       | I2C1                                      |
| PWDN            | 24         | GP18        | Power Down      | Active High or Low (check module)         |
| RESET           | 25         | GP19        | Reset           | Active Low usually                        |
| XCLK            | 22         | GP17        | System Clock    | Needs PWM output capability               |
| PCLK            | 19         | GP14        | Pixel Clock     | Input to RP2040                           |
| VSYNC           | 20         | GP15        | Vertical Sync   | Input to RP2040                           |
| HSYNC/HREF      | 21         | GP16        | Horizontal Sync | Input to RP2040                           |
| D0              | 9          | GP6         | Data Bit 0      | Parallel Data Input                       |
| D1              | 10         | GP7         | Data Bit 1      | Parallel Data Input                       |
| D2              | 11         | GP8         | Data Bit 2      | Parallel Data Input                       |
| D3              | 12         | GP9         | Data Bit 3      | Parallel Data Input                       |
| D4              | 14         | GP10        | Data Bit 4      | Parallel Data Input                       |
| D5              | 15         | GP11        | Data Bit 5      | Parallel Data Input                       |
| D6              | 16         | GP12        | Data Bit 6      | Parallel Data Input                       |
| D7              | 17         | GP13        | Data Bit 7      | Parallel Data Input                       |
| **Buck-Boost**  |            |             |                 |                                           |
| EN (Enable)     | 29         | GP22        | Buck-Boost EN   | Connect to TPS63031 EN pin              |

