import threading
import logging

log = logging.getLogger("global-store")

# Global Store to maintain application-wide state
# =========== Value Definitions ===========
# "wifi_status": str - Current WiFi connection status ("CONNECTED", "DISCONNECTED", etc.)
# "is_ap_mode": bool - Whether the device is in Access Point mode
# "is_sta_mode": bool - Whether the device is in Station mode
# "is_have_wifi_connect_signal": bool - Whether is have request to connect WiFi
# "camera_status": str - Whether the camera is running properly ("INACTIVE", "RUNNING", "RESTARTING" ,"STOPPED", etc.)
# "retries_count": int - Retry count value for restart camera pipeline

class GlobalStore:
    def __init__(self):
        self._lock = threading.Lock()
        
        self._data = {
            "wifi_status": "DISCONNECTED",
            "is_ap_mode": False,
            "is_sta_mode": True,
            "is_have_wifi_connect_signal": False,
            "camera_status": "INACTIVE",
            "retries_count": 0
        }

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                if self._data[key] != value:
                    log.info(f"Update {key}: {self._data[key]} -> {value}")
                    self._data[key] = value
            else:
                log.warning(f"Key '{key}' not found in GlobalStore.")

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def increment(self, key, amount=1):
        with self._lock:
            if isinstance(self._data.get(key), (int, float)):
                self._data[key] += amount
            else:
                log.error(f"Cannot increment non-numeric key: {key}")

# Global instance
Global_Store = GlobalStore()