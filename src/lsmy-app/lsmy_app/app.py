# =============================================================================
#  LSMY Application Runtime
# -----------------------------------------------------------------------------
#  Package     : lsmy_app
#  Role        : Main application runtime
#
#  Responsibilities:
#   - Implement main application execution flow
#   - Run the primary control loop
#   - Orchestrate system components (sensors, AI, core services)
#   - Interface with native C libraries
#   - Manage threads, tasks, and internal state
#   - Handle application-level errors and graceful shutdown
#
#  Invocation:
#   - Launched by the run-lsmy system entrypoint
#   - Must not be executed directly as a standalone script
#
#  IMPORTANT:
#   - This is the core runtime of the LSMY system
#   - No system/bootstrap logic here
#   - No process-level initialization here
#   - Keep clear separation between runtime logic and entrypoint
# =============================================================================

import asyncio
import sys
import threading
import time
import signal
import ctypes
import logging
from enum import Enum, auto


############################################
# PYTHON LIBRARY FOR LSMY                  #
# - LSMY Hello World Library Example       #
# - Another Library Example                #
############################################

# ====== HELLO WORLD LIBRARY ======
from lsmy_python_lib.hello import say_hello

# ====== WIFI MODE LIBRARY ======
from lsmy_python_lib.wifi_mode_manager import WiFiModeManager

# ====== WIFI CONFIG LIBRARY ======
from lsmy_python_lib.wifi_config_manager import WiFiConfigManager
from lsmy_python_lib.wifi_config_manager import update_wifi_connect_signal

# ====== WEBSERVER LIBRARY ======
from lsmy_webserver.manager import ProvisionWebserverManager

# ====== IPC LIBRARY ======
from lsmy_python_lib.ipc import ipc_server_task

# ====== BUTTON RESET LIBRARY ======
from lsmy_python_lib.button_handler import monitor_button_reset

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

# ====== GLOBAL STORE LIBRARY ======
from lsmy_python_lib.global_store import Global_Store

# ====== ANOTHER LIBRARY ======
# Additional Python library imports can go here


############################################
# C/C++ LIBRARY FOR LSMY LAB MONITORING    #
# - LSMY Hello World Library Example       #
# - Another Library Example                #
############################################

# ====== HELLO WORLD LIBRARY ======
lib = ctypes.CDLL("/usr/lib/liblsmy_hello.so.1")
lib.hello_print.restype = None
lib.hello_print.argtypes = []

# ====== ANOTHER LIBRARY ======
# Additional C/C++ library imports can go here


# -------------------------
# Logging Configuration
# -------------------------
log = logging.getLogger("lsmy-app")


# -------------------------
# Application State
# -------------------------
class AppState(Enum):
    """
    Application lifecycle states.
    """
    INIT = auto()       # Application created, not running yet
    RUNNING = auto()    # Main loop active
    ERROR = auto()      # Fatal error occurred
    STOPPED = auto()    # Application fully stopped


class LsmyApplication:
    """
    LSMY Application
    Controls application lifecycle and main execution flow.
    """

    #-------- Constructor --------
    def __init__(self):
        self.state = AppState.INIT
        # WifiModeManager
        self.wifi_manager = WiFiModeManager()
        self.wifi_config_manager = WiFiConfigManager()
        self.provision_webserver_manager = ProvisionWebserverManager()

        self.print_wifi_info = False
        self.running = False

    # -------- Public lifecycle --------
    def start(self):
        self._setup_signal_handlers()
        self._startup_sequence()
        self._main_loop()

    def stop(self):
        self._shutdown_sequence()

    # -------- Startup / shutdown --------
    def _startup_sequence(self):
        log.info("############################################")
        log.info("#   LSMY SYSTEM - STARTUP                  #")
        log.info("############################################")

        self._load_configuration()
        self._initialize_services()

        self.state = AppState.RUNNING
        self.running = True

        log.info("LSMY system startup completed")

    def _shutdown_sequence(self):
        log.info("############################################")
        log.info("#   LSMY SYSTEM - SHUTDOWN                 #")
        log.info("############################################")

        self.running = False
        self._stop_services()
        self.state = AppState.STOPPED

        log.info("LSMY system shutdown completed")

    # -------- Initialization --------
    def _load_configuration(self):
        log.info("Loading system configuration")
        pass

    def _initialize_services(self):
        log.info("Initializing core services")
        self._init_sensor_subsystem()
        self._init_ai_subsystem()
        self._init_communication_subsystem()

    def _stop_services(self):
        log.info("Stopping core services")

        log.info("Stopping wifi mode services")
        self.wifi_manager.cleanup_wifi()
        log.info("Stopping provision webserver services")
        self.provision_webserver_manager.stop()

        log.info("Stopping network time synchronization services")
        run_cmd(["systemctl", "stop", "wpa_supplicant"], check=False)
        
        log.info("Core services stopped")

    # -------- Signals --------
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)

    def _handle_termination(self, signum, frame):
        log.info(f"Received termination signal ({signum})")
        self.stop()

    # -------- Main loop --------
    def _main_loop(self):
        log.info("========== ENTERING MAIN APPLICATION LOOP ==========")

        # Monitor Reset Button Thread
        monitor_button_reset_thread = threading.Thread(target=monitor_button_reset, args=(self.wifi_manager,), daemon=True)
        monitor_button_reset_thread.start()

        # IPC Server Thread
        ipc_thread = threading.Thread(target=self.start_ipc_thread, daemon=True)
        ipc_thread.start()

        while self.running:
            # Main logic connections here
            if self.wifi_manager.is_wifi_connected():
                # Wifi connected, connect to CoreIoT

                if self.print_wifi_info:
                    self.print_wifi_info = False
                    wifi_info = self.wifi_config_manager.get_wifi_status_iw("wlan0")
        
                    if wifi_info:
                        log.info("========== WIFI CONNECTED ==========")
                        log.info(f"SSID      : {wifi_info.get('ssid')}")
                        log.info(f"IP Addr   : {wifi_info.get('ip')}")
                        log.info(f"Signal    : {wifi_info.get('signal')}")
                        log.info("====================================")
                    else:
                        log.info("WiFi connected, but could not retrieve detailed info.")

                    Global_Store.set("wifi_status", "CONNECTED")
                else:
                    log.info("WiFi connected, system operational")
            else:
                # Wifi not connected
                wifi_mode = self.wifi_manager.get_wifi_role()
                log.info(f"Current WiFi mode: {wifi_mode}")

                if wifi_mode == "STA":
                    # In STA mode but not connected, first try to connection
                    if self.wifi_config_manager.has_any_wifi_config():
                        log.info("Attempting to connect to WiFi in STA mode")
                        self.wifi_manager.switch_to_sta()
                        self.wifi_manager.start_sta_services()

                        log.info(f"Waiting for wlan0 to connect...")
                        if self.wifi_config_manager.is_wait_for_wifi():
                            self.wifi_config_manager.request_ip(interface="wlan0")
                            self.print_wifi_info = True
                        else:
                            log.info("WiFi connection failed, switching to AP mode")
                            self.wifi_manager.switch_to_ap()
                            self.provision_webserver_manager.start() 
                    else:
                        log.info("No WiFi config found, switching to AP mode")
                        self.wifi_manager.switch_to_ap()
                        self.provision_webserver_manager.start()
                elif wifi_mode == "AP":
                    # Check if have any wifi config
                    is_have_wifi_connect = self.wifi_config_manager.get_wifi_connect_signal()
                    log.info(f"Is have WiFi connect: {is_have_wifi_connect}")

                    # In AP mode, stay in AP mode
                    if (not self.provision_webserver_manager.is_running()) and (not is_have_wifi_connect):
                        log.info("Provisioning webserver not running, starting it")
                        self.wifi_manager.switch_to_ap()
                        self.provision_webserver_manager.start()
                    else:
                        log.info("Staying in AP mode, waiting for user configuration")

                        # If have update wifi connect signal, switch to STA mode
                        if is_have_wifi_connect:
                            log.info("WiFi connect signal found, switching to STA mode")
                            self.wifi_manager.switch_to_sta()
                            self.provision_webserver_manager.stop()
                            update_wifi_connect_signal(False)
                else:
                    log.warning("Unknown WiFi mode, switching to STA mode")
                    self.wifi_manager.cleanup_wifi()
                    self.provision_webserver_manager.stop()

            self._run_cycle()
            time.sleep(5)

    def _run_cycle(self):
        """
        Single application cycle.
        Placeholder for sensor polling, AI inference, and data publishing.
        """
        # Call the function from the shared library
        # lib.hello_print()

        # Call the Python function
        # say_hello()

        pass

    # -------- Application Thread --------
    def start_ipc_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ipc_server_task())

    # -------- Subsystems --------
    def _init_sensor_subsystem(self):
        log.info("Initializing sensor subsystem")
        pass

    def _init_ai_subsystem(self):
        log.info("Initializing AI subsystem")
        pass

    def _init_communication_subsystem(self):
        log.info("Initializing communication subsystem")
        pass
