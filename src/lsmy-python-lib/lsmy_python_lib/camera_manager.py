import time
import logging
import threading

# ====== GLOBAL STORE LIBRARY ======
from lsmy_python_lib.global_store import Global_Store

log = logging.getLogger("camera-manager")

MAX_RECOVER_TRIES = 3

class CameraManager:
    """
    Camera Manager to monitor and restart camera
    """

    def __init__(self, device_id=0):
        log.info("CameraManager initialized")
        self.device_id = device_id
        self.max_recover_retries = MAX_RECOVER_TRIES
        self._stop_event = threading.Event()

        update_camera_status("INACTIVE")


    def start(self):
        """
        Start camera manager thread
        """
        log.info("Camera manager successfully started")

    def stop(self):
        """
        Stop camera manager thread
        """
        self._stop_event.set()

    def camera_main_process(self):
        log.info("Starting Camera main process thread...")
        update_camera_status("RUNNING")

        retries_count = 0

        while True:
            camera_status = ""
            while not self._stop_event.is_set():
                try:
                    camera_status = get_camera_status()

                    if(camera_status == "STOPPED"):
                        self._stop_event.set()
                        continue
                    elif(camera_status == "RESTARTING"):
                        if retries_count < self.max_recover_retries:
                            self._stop_event.set()
                            retries_count = retries_count + 1
                            continue
                        else:
                            retries_count = 0
                            update_camera_status("STOPPED")
                            continue
                    else:
                        self._stop_event.wait(5)
                        continue
                    # self._stop_event.wait(1)

                except Exception as e:
                    log.error(f"Camera main process encountered an error: {e}")
                    self._stop_event.wait(5)

            # Clean actions
            try:
                # TODO: stop streaming
                # TODO: release buffers
                # TODO: close device
                pass
            except Exception as e:
                log.error(f"Error while stopping camera pipeline: {e}")
            
            if(camera_status == "STOPPED"):
                break
            elif(camera_status == "RESTARTING"):
                self._stop_event.clear()

        update_camera_status("STOPPED")

# Update camera_status
def update_camera_status(value: str):
    Global_Store.set("camera_status", value)

# Get camera_status
def get_camera_status():
    Global_Store.get("camera_status")
        