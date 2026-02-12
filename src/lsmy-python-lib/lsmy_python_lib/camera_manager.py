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
        self._max_recover_retries = MAX_RECOVER_TRIES
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
        log.info("========== STOPPING CAMERA MAIN PROCESS THREAD ==========")
        update_camera_status("STOPPED")
        self._stop_event.set()

    def camera_main_process(self):
        log.info("========== STARTING CAMERA MAIN PROCESS THREAD ==========")
        update_camera_status("INACTIVE")

        camera_status = ""
        while True:
            while not self._stop_event.is_set():
                try:
                    camera_status = get_camera_status()

                    if(camera_status == "INACTIVE"):
                        self._stop_event.wait(5)
                        # TODO: start pipeline, clean actions
                    elif(camera_status == "RUNNING"):
                        update_retries_count(0)
                        self._stop_event.wait(5)
                        # TODO: start streaming
                    elif(camera_status == "STOPPED"):
                        self._stop_event.set()
                        continue
                    elif(camera_status == "RESTARTING"):
                        retries_count = get_retries_count()
                        if retries_count < self._max_recover_retries:
                            self._stop_event.set()
                            retries_count = retries_count + 1
                            update_retries_count(retries_count)
                        else:
                            update_camera_status("INACTIVE")
                            self._stop_event.set()
                        self._stop_event.wait(5)
                    else:
                        log.info(f"Camera status unknown: {camera_status}")
                        update_camera_status("INACTIVE")
                        self._stop_event.set()
                        self._stop_event.wait(5)

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
            
            camera_status = get_camera_status()
            if(camera_status == "STOPPED"):
                break
            elif(camera_status == "RESTARTING" or camera_status == "INACTIVE"):
                self._stop_event.clear()

        log.info("Camera stopped running process!!!")

# Update camera_status
def update_camera_status(value: str):
    Global_Store.set("camera_status", value)

# Get camera_status
def get_camera_status():
    return Global_Store.get("camera_status")

# Update retries_count
def update_retries_count(value: int):
    Global_Store.set("retries_count", value)
    
# Get retries_count
def get_retries_count():
    return Global_Store.get("retries_count")
        