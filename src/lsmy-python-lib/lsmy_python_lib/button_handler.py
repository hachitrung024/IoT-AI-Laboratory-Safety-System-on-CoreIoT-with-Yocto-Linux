from datetime import timedelta
import time
import logging
import multiprocessing
import gpiod
from gpiod.line import Direction, Edge, Bias, Value

# ====== WIFI MODE LIBRARY ======
from lsmy_python_lib.wifi_mode_manager import WiFiModeManager

# ====== WIFI CONFIG LIBRARY ======
from lsmy_python_lib.wifi_config_manager import reset_wifi_config

log = logging.getLogger("button-handler")

CHIP_PATH = '/dev/gpiochip0'
BUTTON_PIN = 17

class ResetButtonManager:
    def __init__(self, stop_signal, ready_signal):
        log.info("ResetButtonManager initialized")
        self._stop_event = stop_signal
        self._ready_event = ready_signal

    def start(self):
        """
        Start button handle reset
        """
        log.info("Reset button manager successfully started")

    def stop(self):
        """
        Stop button handle process
        """
        log.info("========== STOPPING RESET BUTTON MONITOR PROCESS ==========")
        self._stop_event.set()
        self._ready_event.clear()

    def execute_full_reset(self, wifi_manager: WiFiModeManager):
        log.warning("!!! STARTING FACTORY RESET !!!")

        # Clear WiFi configurations
        reset_wifi_config()
        # Disconnect WiFi
        wifi_manager.cleanup_wifi()

        log.info("Reset complete. WiFi disconnected and Config cleared.")

    def monitor_button_reset(self, wifi_manager: WiFiModeManager):
        log.info("========== STARTING RESET BUTTON MONITOR PROCESS ==========")

        try:
            line_settings = gpiod.LineSettings(
                direction=Direction.INPUT,
                edge_detection=Edge.FALLING,
                bias=Bias.PULL_UP,
                debounce_period=timedelta(milliseconds=50)
            )

            with gpiod.request_lines(
                CHIP_PATH,
                consumer="reset-button-monitor",
                config={BUTTON_PIN: line_settings}
            ) as request:
                
                press_start = 0
                log.info(f"Button monitor active on GPIO {BUTTON_PIN} (gpiod)")
                self._ready_event.set()

                while not self._stop_event.is_set():
                    if request.wait_edge_events(timedelta(seconds=5)):
                        request.read_edge_events()
                        
                        press_start = time.time()
                        log.info("Button Pressed")

                        while not self._stop_event.is_set():
                            current_state = request.get_value(BUTTON_PIN)

                            if current_state == Value.INACTIVE:
                                if self._stop_event.wait(0.1): 
                                    break
                            else:
                                final_duration = time.time() - press_start
                                log.info(f"Button Released. Total duration: {final_duration:.2f}s")
                                
                                if final_duration >= 3.0:
                                    log.warning("Full reset triggered!")
                                    self.execute_full_reset(wifi_manager)
                                
                                break

                # Clean actions

        except Exception as e:
            log.error(f"gpiod Monitor Error: {e}")