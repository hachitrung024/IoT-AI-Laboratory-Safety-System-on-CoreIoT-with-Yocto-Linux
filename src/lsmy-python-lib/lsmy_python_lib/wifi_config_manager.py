import os
import re
import logging
import subprocess
import time
from typing import List, Dict

# ====== GLOBAL STORE LIBRARY ======
from lsmy_python_lib.global_store import Global_Store

log = logging.getLogger("wifi-config")

WPA_CONF = "/etc/wpa_supplicant.conf"

HEADER_LINES = [
    "ctrl_interface=/var/run/wpa_supplicant",
    "ctrl_interface_group=0",
    "update_config=1",
    "country=VN",
]

class WiFiConfigManager:
    def __init__(self):
        log.info("WiFiConfigManager initialized")

    # Get is_have_wifi_connect signal
    def get_wifi_connect_signal(self) -> bool:
        return Global_Store.get("is_have_wifi_connect_signal")

    # Load WiFi configurations function
    def load_wifi_configs(self, config_path: str = WPA_CONF) -> List[Dict]:
        """
        Load all WiFi network blocks from wpa_supplicant.conf
        """
        if not os.path.exists(config_path):
            return []

        with open(config_path, "r") as f:
            content = f.read()

        networks = re.findall(r'network\s*=\{.*?\n\}', content, re.DOTALL)
        results = []

        for net in networks:
            ssid_match = re.search(r'ssid\s*=\s*"(.+?)"', net)
            psk_match = re.search(r'psk\s*=\s*"(.+?)"', net)

            if not ssid_match:
                continue

            results.append({
                "ssid": ssid_match.group(1),
                "password": psk_match.group(1) if psk_match else None,
                "raw": net.strip(),
            })

        return results

    # Check if WiFi config exists function
    def has_any_wifi_config(self, config_path: str = WPA_CONF) -> bool:
        configs = self.load_wifi_configs(config_path)
        return len(configs) > 0

    # Wait for WiFi connection
    def is_wait_for_wifi(self, interface="wlan0", timeout=10) -> bool:
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = subprocess.run(["wpa_cli", "-i", interface, "status"], 
                                    capture_output=True, text=True)
            status_output = result.stdout
            
            if "wpa_state=COMPLETED" in status_output:
                log.info("WiFi is connected successfully!")
                return True
            
            if "wpa_state=DISCONNECTED" in status_output:
                log.info("WiFi disconnected, retrying...")
                pass
                
            time.sleep(1)

        log.error("WiFi connection timeout!")
        return False
    
    def request_ip(self, interface="wlan0"):
        log.info("Requesting IP address for %s...", interface)
        subprocess.run(["udhcpc", "-i", interface, "-n", "-q"], check=False)

    def get_wifi_status_iw(self, iface="wlan0"):
        info = {"connected": False, "ssid": None, "ip": None, "signal": None}
        
        try:
            res_link = subprocess.run(["iw", "dev", iface, "link"], capture_output=True, text=True)
            link_output = res_link.stdout

            if "Not connected" not in link_output and "SSID" in link_output:
                info["connected"] = True
                # SSID and signal
                ssid_match = re.search(r"SSID:\s+(.*)", link_output)
                signal_match = re.search(r"signal:\s+(-\d+\s+dBm)", link_output)
                
                if ssid_match: info["ssid"] = ssid_match.group(1).strip()
                if signal_match: info["signal"] = signal_match.group(1).strip()

                # Ip address
                res_ip = subprocess.run(["ip", "-4", "addr", "show", iface], capture_output=True, text=True)
                ip_match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", res_ip.stdout)
                if ip_match:
                    info["ip"] = ip_match.group(1)

            return info if info["connected"] else None

        except Exception as e:
            log.error(f"Fault using iw: {e}")
            return None

# Update is_have_wifi_connect signal
def update_wifi_connect_signal(value: bool):
    Global_Store.set("is_have_wifi_connect_signal", value)
    # log.info("WiFi config signal updated: %s", IS_HAVE_WIFI_CONNECT_SIGNAL)

# Configure WiFi function
def configure_wifi(ssid, password):
    """
    Save or update WiFi configuration in wpa_supplicant.conf
    Automatically handles multiple networks.
    """
    log.info("Processing WiFi config for: %s", ssid)

    if password:
        new_network = f'network={{\n    ssid="{ssid}"\n    psk="{password}"\n}}'
    else:
        new_network = f'network={{\n    ssid="{ssid}"\n    key_mgmt=NONE\n}}'
    
    existing_networks = []
    
    if os.path.exists(WPA_CONF):
            with open(WPA_CONF, "r") as f:
                content = f.read()
                
            networks = re.findall(r'network\s*=\{.*?\n\}', content, re.DOTALL)
            
            for net in networks:
                if not re.search(f'ssid\s*=\s*"{re.escape(ssid)}"', net):
                    existing_networks.append(net.strip())
                else:
                    log.info("Found old config for '%s', replacing it...", ssid)

    existing_networks.append(new_network)

    with open(WPA_CONF, "w") as f:
        f.write("\n".join(HEADER_LINES) + "\n\n")
        f.write("\n\n".join(existing_networks) + "\n")

    log.info("Saved WiFi '%s' successfully. Total networks remembered: %d", 
            ssid, len(existing_networks))

# Reset WiFi configuration function   
def reset_wifi_config():
    log.info("Resetting WiFi configuration to default...")

    default_content = (
        "ctrl_interface=/var/run/wpa_supplicant\n"
        "ctrl_interface_group=0\n"
        "update_config=1\n"
        "country=VN\n\n"
        "network={\n"
        "    key_mgmt=NONE\n"
        "}\n"
    )

    try:
        with open(WPA_CONF, "w") as f:
            f.write(default_content)
        
        log.info("WiFi configuration has been reset successfully.")
        
        return True
    except Exception as e:
        log.error("Failed to reset WiFi config: %s", e)
        return False