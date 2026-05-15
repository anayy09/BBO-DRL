"""
Hardware profiles based on published datasheets.
ESP32-S3: https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf
Raspberry Pi 4: https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    cpu_freq_hz: float        # cycles per second
    kappa: float              # effective switched capacitance (F/cycle^2), CMOS model
    tx_power_w: float         # transmission power in watts (0 if not wireless transmitter)
    idle_power_w: float       # idle/listening power in watts
    ram_bytes: int            # available RAM
    bandwidth_hz: float       # channel bandwidth
    max_mips: float           # compute capacity in millions of instructions per second


# ---------------------------------------------------------------------------
# ESP32-S3: 240 MHz dual-core Xtensa LX7, Wi-Fi 802.11b/g/n
# Datasheet: ESP32-S3 Technical Reference Manual v1.2
#   - Active current at 240 MHz: ~100 mA @ 3.3 V → ~330 mW (conservative)
#   - Wi-Fi TX max: 22.5 dBm ≈ 178 mW (per datasheet, section 5.2)
#   - Modem-sleep idle: ~10 mA @ 3.3 V = 33 mW
#   - Channel bandwidth: 20 MHz (802.11n HT20)
# ---------------------------------------------------------------------------
WEARABLE_ESP32 = HardwareProfile(
    name="ESP32-S3 Wearable",
    cpu_freq_hz=240e6,
    kappa=1e-27,              # typical CMOS effective capacitance for 40-nm process
    tx_power_w=0.178,         # 22.5 dBm = 10^(22.5/10) mW ≈ 178 mW (Wi-Fi max)
    idle_power_w=0.033,       # 10 mA × 3.3 V = 33 mW (modem-sleep listening)
    ram_bytes=512 * 1024,     # 512 KB SRAM
    bandwidth_hz=20e6,        # 20 MHz Wi-Fi HT20 channel
    max_mips=240.0,           # single-core effective MIPS ≈ clock (simple IPC=1 model)
)

# ---------------------------------------------------------------------------
# Raspberry Pi 4 Model B: 1.5 GHz Cortex-A72, BCM2711
# Datasheet / measured power:
#   - Idle draw: ~3.4 W (measured, RPi Foundation)
#   - Gateway receives; wearable transmits → tx_power_w = 0
#   - 5G NR channel (via USB dongle): 100 MHz bandwidth
# ---------------------------------------------------------------------------
EDGE_GATEWAY_RPI4 = HardwareProfile(
    name="Raspberry Pi 4 Edge Gateway",
    cpu_freq_hz=1500e6,
    kappa=1e-28,              # 28-nm CMOS, lower kappa than ESP32
    tx_power_w=0.0,           # gateway is the receiver; wearable pays TX energy
    idle_power_w=3.4,         # measured idle draw at 3.4 W (RPi4 with 8 GB)
    ram_bytes=8 * 1024 * 1024 * 1024,   # 8 GB LPDDR4
    bandwidth_hz=100e6,       # 5G NR sub-6 GHz, 100 MHz channel
    max_mips=1500.0,
)

# ---------------------------------------------------------------------------
# Mid-tier fog compute node: server-grade ARM or x86, ~2.2 GHz
# Modelled after an NVIDIA Jetson AGX Orin / small rack server
# ---------------------------------------------------------------------------
FOG_NODE = HardwareProfile(
    name="Fog Compute Node",
    cpu_freq_hz=2200e6,
    kappa=1e-28,
    tx_power_w=0.0,           # fog node is receiver
    idle_power_w=5.0,         # ~5 W idle for compact server
    ram_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
    bandwidth_hz=100e6,       # 5G NR backhaul
    max_mips=2200.0,
)

# ---------------------------------------------------------------------------
# Cloud server: abstracted as unlimited capacity resource
# Energy not charged to the wearable (metered by cloud provider separately)
# Modelled as a 3.2 GHz instance with 1 Gbps NIC
# ---------------------------------------------------------------------------
CLOUD_SERVER = HardwareProfile(
    name="Cloud Server",
    cpu_freq_hz=3200e6,
    kappa=0.0,                # cloud energy not attributed to wearable budget
    tx_power_w=0.0,
    idle_power_w=0.0,         # abstracted; not charged to IoT device
    ram_bytes=int(1e12),      # effectively unlimited (1 TB)
    bandwidth_hz=1e9,         # 1 Gbps NIC
    max_mips=3200.0,
)
