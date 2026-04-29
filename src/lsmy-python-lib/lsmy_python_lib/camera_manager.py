import time
import logging
import multiprocessing

# ====== GLOBAL STORE LIBRARY ======
from lsmy_python_lib.global_store import GlobalStore

from lsmy_python_lib.image_analytics_engine import ImageAnalyticsEngine, MODEL_BLAZE_FACE_DETECTION, MODEL_LANDMARK_FACE_DETECTION

log = logging.getLogger("camera-manager")

FPS = 15
MAX_RECOVER_TRIES = 3

class CameraManager:
    """
    Camera Manager to monitor and restart camera
    """

    def __init__(self, global_store, stop_signal, ready_signal, device_id=0):
        log.info("CameraManager initialized")
        self.global_store = global_store
        self.device_id = device_id
        self._max_recover_retries = MAX_RECOVER_TRIES
        self._stop_event = stop_signal
        self._ready_event = ready_signal

        update_camera_status(self.global_store,"INACTIVE")


    def start(self):
        """
        Start camera manager process
        """
        log.info("Camera manager successfully started")

    def stop(self):
        """
        Stop camera manager process
        """
        log.info("========== STOPPING CAMERA MAIN PROCESS ==========")
        update_camera_status(self.global_store,"STOPPED")
        self._stop_event.set()
        self._ready_event.clear()

    def camera_main_process(self):
        log.info("========== STARTING CAMERA MAIN PROCESS ==========")
        update_camera_status(self.global_store,"INACTIVE")
        self._ready_event.set()

        camera_status = ""

        engine = ImageAnalyticsEngine(width=640, height=480, fps=FPS,
                               model_blaze_path=MODEL_BLAZE_FACE_DETECTION, model_landmark_path=MODEL_LANDMARK_FACE_DETECTION,
                               use_model=True, debug_mode=True)
        while True:
            while not self._stop_event.is_set():
                try:
                    camera_status = get_camera_status(self.global_store)

                    if(camera_status == "INACTIVE"):
                        self._stop_event.wait(5)
                    elif(camera_status == "RUNNING"):
                        if get_retries_count(self.global_store) != 0:
                            update_retries_count(self.global_store,0)
        
                        # Start camera pipeline
                        try:
                            engine.start()
                            
                            with engine.critical_lock:
                                if engine.is_critical:
                                    log.info("Critical status detected, stopping engine...")
                                    engine.stop()
                                    break
                        except KeyboardInterrupt:
                            log.info("Interrupted by keyboard")
                        except Exception as e:
                            log.exception("Unexpected error occurred: %s", e)
                        self._stop_event.wait(5)
                    elif(camera_status == "STOPPED"):
                        self._stop_event.set()
                        continue
                    elif(camera_status == "RESTARTING"):
                        retries_count = get_retries_count(self.global_store)
                        if retries_count < self._max_recover_retries:
                            self._stop_event.set()
                            retries_count = retries_count + 1
                            update_retries_count(self.global_store,retries_count)
                        else:
                            update_camera_status(self.global_store,"INACTIVE")
                            self._stop_event.set()
                        time.sleep(5)
                    else:
                        log.info(f"Camera status unknown: {camera_status}")
                        update_camera_status(self.global_store,"INACTIVE")
                        self._stop_event.set()
                        self._stop_event.wait(5)

                except Exception as e:
                    log.error(f"Camera main process encountered an error: {e}")
                    self._stop_event.wait(5)

            # Clean actions
            try:
                if engine.running:
                    engine.stop()
                else:
                    log.info("Skipping stop as engine is not running")
            except Exception as e:
                log.error(f"Error while stopping camera pipeline: {e}")
            
            camera_status = get_camera_status(self.global_store)
            if(camera_status == "STOPPED"):
                break
            elif(camera_status == "RESTARTING" or camera_status == "INACTIVE"):
                self._stop_event.clear()

        log.info("Camera stopped running process!!!")

# Update camera_status
def update_camera_status(global_store: GlobalStore, value: str):
    global_store.set("camera_status", value)

# Get camera_status
def get_camera_status(global_store: GlobalStore):
    return global_store.get("camera_status")

# Update retries_count
def update_retries_count(global_store: GlobalStore, value: int):
    global_store.set("retries_count", value)
    
# Get retries_count
def get_retries_count(global_store: GlobalStore):
    return global_store.get("retries_count")
        