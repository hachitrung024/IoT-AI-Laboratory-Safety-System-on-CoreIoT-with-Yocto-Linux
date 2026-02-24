import time
import subprocess
import logging
from enum import Enum

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

# ====== GLOBAL STORE LIBRARY ======
from lsmy_python_lib.global_store import GlobalStore

log = logging.getLogger("wifi-mode")

SYSTEMD_NETWORK_FILE = "/etc/systemd/network/10-wlan0.network"
AP_NETWORK_FILE = "/etc/systemd/network/wlan0-ap.network"
STA_NETWORK_FILE = "/etc/systemd/network/wlan0-sta.network"


class WiFiMode(Enum):
    AP = "ap"
    STA = "sta"


class WiFiModeManager:
    def __init__(self, global_store):
        self.mode = WiFiMode.STA
        self.global_store = global_store
        log.info("WiFiModeManager initialized with mode=%s", self.mode.value)

    def _link_network(self, target: str):
        run_cmd([
            "ln", "-sf", target, SYSTEMD_NETWORK_FILE
        ])

    def _restart_networkd(self):
        run_cmd(["systemctl", "restart", "systemd-networkd"])

    def _wait_for_interface(self, iface: str, timeout: int = 10):
        log.info("Waiting for interface %s...", iface)

        for _ in range(timeout):
            result = subprocess.run(
                ["ip", "link", "show", iface],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                log.info("%s is present", iface)
                return True

            time.sleep(1)

        raise TimeoutError(f"Interface {iface} not available")


    def switch_to_ap(self):
        log.info("========== SWITCH TO AP MODE ==========")
        # self.mode = WiFiMode.AP

        self._link_network(AP_NETWORK_FILE)
        self._restart_networkd()

        run_cmd(["systemctl", "stop", "wpa_supplicant"])
        self._wait_for_interface("wlan0", timeout=10)
        run_cmd_with_retry(["systemctl", "start", "hostapd"], retries=5, delay=2)
        run_cmd_with_retry(["systemctl", "start", "dnsmasq"], retries=5, delay=2)

        log.info("AP mode enabled")

        self.global_store.set("wifi_status", "DISCONNECTED")
        self.global_store.set("is_ap_mode", True)
        self.global_store.set("is_sta_mode", False)

    def switch_to_sta(self):
        log.info("========== SWITCH TO STA MODE ==========")
        # self.mode = WiFiMode.STA

        self._link_network(STA_NETWORK_FILE)
        self._restart_networkd()

        run_cmd(["systemctl", "stop", "hostapd"])
        run_cmd(["systemctl", "stop", "dnsmasq"])
        run_cmd(["systemctl", "enable", "wpa_supplicant"])

        log.info("STA mode enabled")

        self.global_store.set("wifi_status", "DISCONNECTED")
        self.global_store.set("is_ap_mode", False)
        self.global_store.set("is_sta_mode", True)

    def start_sta_services(self):
        log.info("Starting STA services...")
        run_cmd(["systemctl", "start", "wpa_supplicant"])

    # Cleanup function to reset WiFi state
    def cleanup_wifi(self):
        log.info("========== CLEANUP WIFI STATE ==========")

        # Stop AP-related services
        run_cmd(["systemctl", "stop", "hostapd"], check=False)
        run_cmd(["systemctl", "stop", "dnsmasq"], check=False)

        # Stop STA service
        run_cmd(["systemctl", "stop", "wpa_supplicant"], check=False)
        # Restore default network config (STA)
        self._link_network(STA_NETWORK_FILE)

        # Restart network stack
        self._restart_networkd()

        # self.mode = WiFiMode.STA
        log.info("WiFi state cleaned, system returned to STA baseline")

        self.global_store.set("wifi_status", "DISCONNECTED")
        self.global_store.set("is_ap_mode", False)
        self.global_store.set("is_sta_mode", True)

    # Get current WiFi role
    def get_wifi_role(self, iface: str = "wlan0") -> str:
        # Detect WiFi role using iw:
        # - STA  -> type managed
        # - AP   -> type AP
        result = subprocess.run(
            ["iw", "dev", iface, "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            log.warning("iw failed: %s", result.stderr.strip())
            return "UNKNOWN"

        output = result.stdout

        if "type AP" in output:
            return "AP"
        if "type managed" in output:
            return "STA"

        return "UNKNOWN"

    # Check if interface has default ip
    def has_ip(self, iface: str = "wlan0") -> bool:
        # Check if interface has an IPv4 address
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return False

        return "inet " in result.stdout

    # Check if interface has default route
    def has_default_route(self, iface: str = "wlan0") -> bool:
        # Check if default route exists on interface
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return False

        return iface in result.stdout

    # Check is wifi is connected
    def is_wifi_connected(self, iface: str = "wlan0") -> bool:
        # STA mode + has IP + Iw dev has default route => WiFi usable
        iw_dev_result = False
        result = subprocess.run(
                ["iw", "dev", iface, "link"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
        
        if "Connected to" in result.stdout:
            iw_dev_result = True
        else: 
            iw_dev_result = False

        return (
            self.get_wifi_role() == "STA"
            and self.has_ip("wlan0")
            and self.has_default_route("wlan0")
            and iw_dev_result
        )

