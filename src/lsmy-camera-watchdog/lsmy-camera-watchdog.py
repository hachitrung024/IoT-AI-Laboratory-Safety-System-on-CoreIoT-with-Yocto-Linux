#!/usr/bin/python3
import os
import sys
import time
import logging
import asyncio

# ====== IPC LIBRARY ======
from lsmy_python_lib.ipc import send_update_camera_status_signal_ipc

# ===== CONFIG =====
CAMERA_DEVICE = "/dev/camera0"
CHECK_INTERVAL = 20           
MAX_INACTIVE_TIME = 20
MAX_RECOVER_TRIES = 3

# ===== STATE =====
last_interrupt_count = None
inactive_duration = 0
is_retry = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("camera-watchdog")


# ---------- LOW LEVEL CHECK ----------
def get_unicam_interrupt_count():
    with open("/proc/interrupts", "r") as f:
        for line in f:
            if "unicam" in line:
                return int(line.split()[1])
    return None


# ---------- HEALTH CHECK ----------
def check_camera_health():
    global last_interrupt_count, inactive_duration

    if not os.path.exists(CAMERA_DEVICE):
        log.error("Camera device not found")
        return False

    current = get_unicam_interrupt_count()
    if current is None:
        log.error("Unicam interrupt not found")
        return False

    if last_interrupt_count is None:
        last_interrupt_count = current
        return True

    if current <= last_interrupt_count:
        inactive_duration += CHECK_INTERVAL
        log.warning(f"Camera inactive for {inactive_duration}s")
    else:
        inactive_duration = 0
        last_interrupt_count = current
        log.debug("Camera active")

    return inactive_duration < MAX_INACTIVE_TIME


# ---------- RECOVERY ----------
async def recover_camera():
    log.warning("Trying to recover camera...")

    try:
        is_retry = True

        # Option 1: restart camera pipeline
        data = {
            "status": "RESTARTING" 
        }
        resq = await send_update_camera_status_signal_ipc(data)

        if resq.get("status") == "ok":
            retries_count = resq["data"]["retries_count"]

            if retries_count < MAX_RECOVER_TRIES:
                is_retry = True
            else:
                is_retry = False
        else:
            log.error("IPC Server returned error: %s", resq.get("error"))
            return False

        # Option 2: reload driver (do not recommneded)
        # subprocess.run(["modprobe", "-r", "unicam"], check=False)
        # subprocess.run(["modprobe", "unicam"], check=False)

        return is_retry
    except Exception as e:
        log.error(f"Recover failed: {e}")
        return False


# ---------- MAIN LOOP ----------
async def main():
    global inactive_duration, is_retry

    while True:
        healthy = check_camera_health()

        if not is_retry:
            log.warning("Recovery failed, but watchdog still running...")

            if healthy:
                log.warning("Camera recovered, try to start camera pipeline...")
                is_retry = True
                data = {
                    "status": "RUNNING" 
                }
                await send_update_camera_status_signal_ipc(data)

            await asyncio.sleep(CHECK_INTERVAL)
            continue

        if not healthy:
            log.error(f"Camera unhealthy, try to restart camera pipeline...")
            is_retry = await recover_camera()
            inactive_duration = 0

        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
