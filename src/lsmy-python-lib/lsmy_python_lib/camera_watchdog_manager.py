import logging
import subprocess

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

log = logging.getLogger("camera-watchdog-manager")

class CameraWatchdogManager:
    """
    Camera Watchdog Manager to monitor and restart camera services
    """

    CAMERA_WATCHDOG_SERVICE = "camera-watchdog.service"

    def __init__(self):
        log.info("CameraWatchdogManager initialized")

    def _is_active(self, service):
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "active"


    def start(self):
        """
        Start camera watchdog service
        """
        log.info("========== STARTING CAMERA WATCHDOG SERVICES ==========")
        run_cmd_with_retry(
            ["systemctl", "start", self.CAMERA_WATCHDOG_SERVICE]
        )

        if not self.is_running():
            raise RuntimeError("Failed to start camera watchdog service")
        log.info("Camera watchdog service successfully started")

    def stop(self):
        """
        Stop camera watchdog service
        """
        log.info("========== STOPPING CAMERA WATCHDOG SERVICES ==========")
        run_cmd(
            ["systemctl", "stop", self.CAMERA_WATCHDOG_SERVICE],
            check=False,
        )

        log.info("Camera watchdog service successfully stopped")
        
    def restart(self):
        log.info("========== RESTARTING CAMERA WATCHDOG SERVICES ==========")
        run_cmd_with_retry(
            ["systemctl", "restart", self.CAMERA_WATCHDOG_SERVICE]
        )

        log.info("Camera watchdog service successfully restarted")

    def is_running(self):
        """
        Check if camera watchdog is active
        """
        return self._is_active(self.CAMERA_WATCHDOG_SERVICE)