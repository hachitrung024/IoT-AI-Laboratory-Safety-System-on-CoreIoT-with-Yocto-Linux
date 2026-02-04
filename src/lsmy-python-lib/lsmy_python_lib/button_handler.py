import time
import logging
import gpiod
from gpiod.line import Direction, Edge, Bias

# ====== WIFI MODE LIBRARY ======
from lsmy_python_lib.wifi_mode_manager import WiFiModeManager

# ====== WIFI CONFIG LIBRARY ======
from lsmy_python_lib.wifi_config_manager import reset_wifi_config

log = logging.getLogger("button-handler")

CHIP_PATH = '/dev/gpiochip0'
BUTTON_PIN = 17

def execute_full_reset(wifi_manager: WiFiModeManager):
    log.warning("!!! STARTING FACTORY RESET !!!")

    # Clear WiFi configurations
    reset_wifi_config()
    # Disconnect WiFi
    wifi_manager.cleanup_wifi()

    log.info("Reset complete. WiFi disconnected and Config cleared.")

def monitor_button_reset(wifi_manager: WiFiModeManager):
    try:
        line_settings = gpiod.LineSettings(
            direction=Direction.INPUT,
            edge_detection=Edge.BOTH,
            bias=Bias.PULL_UP
        )

        with gpiod.request_lines(
            CHIP_PATH,
            consumer="reset-button-monitor",
            config={BUTTON_PIN: line_settings}
        ) as request:
            
            press_start = 0
            log.info(f"Button monitor active on GPIO {BUTTON_PIN} (gpiod)")

            while True:
                if request.wait_edge_events():
                    for event in request.read_edge_events():
                        if event.event_type == gpiod.EdgeEvent.Type.FALLING_EDGE:
                            press_start = time.time()
                        
                        elif event.event_type == gpiod.EdgeEvent.Type.RISING_EDGE:
                            if press_start > 0:
                                duration = time.time() - press_start
                                log.info(f"Button released after {duration:.2f}s")
                                
                                if duration >= 3.0:
                                    log.warning("Full reset triggered!")
                                    execute_full_reset(wifi_manager)
                                
                                press_start = 0

    except Exception as e:
        log.error(f"gpiod Monitor Error: {e}")