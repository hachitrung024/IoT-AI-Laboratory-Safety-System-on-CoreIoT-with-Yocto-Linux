import time
import logging
import subprocess

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

log = logging.getLogger("provision-webserver-manager")

class ProvisionWebserverManager:
    """
    Manage WiFi provisioning services via systemd
    """

    FRONTEND_SERVICE = "provision-web-frontend.service"
    BACKEND_SERVICE  = "provision-web-backend.service"

    def __init__(self):
        log.info("ProvisionManager initialized")

    def _is_active(self, service):
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "active"


    def start(self):
        """
        Start provisioning UI + backend
        """
        log.info("========== STARTING PROVISIONING WEBSERVER SERVICES ==========")
        run_cmd_with_retry(
            ["systemctl", "start", self.FRONTEND_SERVICE]
        )
        run_cmd_with_retry(
            ["systemctl", "start", self.BACKEND_SERVICE]
        )

        if not self.is_running():
            raise RuntimeError("Failed to start provisioning webserver services")
        log.info("Provisioning webserver successfully started")

    def stop(self):
        """
        Stop provisioning UI + backend
        """
        log.info("========== STOPPING PROVISIONING WEBSERVER SERVICES ==========")
        run_cmd(
            ["systemctl", "stop", self.BACKEND_SERVICE],
            check=False,
        )
        run_cmd(
            ["systemctl", "stop", self.FRONTEND_SERVICE],
            check=False,
        )
        log.info("Provisioning webserver successfully stopped")

    def restart(self):
        log.info("========== RESTARTING PROVISIONING WEBSERVER SERVICES ==========")
        run_cmd_with_retry(
            ["systemctl", "restart", self.FRONTEND_SERVICE]
        )
        run_cmd_with_retry(
            ["systemctl", "restart", self.BACKEND_SERVICE]
        )
        log.info("Provisioning webserver successfully restarted")

    def is_running(self):
        """
        Check if provisioning is active
        """
        return self._is_active(self.FRONTEND_SERVICE) and self._is_active(self.BACKEND_SERVICE)