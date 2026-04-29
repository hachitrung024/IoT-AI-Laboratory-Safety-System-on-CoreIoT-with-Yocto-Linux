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

        self.start_engine = True
        self.engine_process = None

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

    def run_engine(stop_event):
        engine = ImageAnalyticsEngine(width=640, height=480, fps=FPS,
                               model_blaze_path=MODEL_BLAZE_FACE_DETECTION, model_landmark_path=MODEL_LANDMARK_FACE_DETECTION,
                               use_model=True, debug_mode=True)

        try:
            engine.start()

            while not stop_event.is_set():
                with engine.critical_lock:
                    if engine.is_critical:
                        log.info("Critical status detected, stopping engine...")
                        engine.stop()
                        break
                stop_event.wait(5)
        except KeyboardInterrupt:
            log.info("Interrupted by keyboard")
        except Exception as e:
            log.exception("Unexpected error occurred: %s", e)
        finally:
            if engine.running:
                engine.stop()
            else:
                log.info("Skipping stop as engine is not running")

    def camera_main_process(self):
        log.info("========== STARTING CAMERA MAIN PROCESS ==========")
        update_camera_status(self.global_store,"INACTIVE")
        self._ready_event.set()

        camera_status = ""

        while True:
            while not self._stop_event.is_set():
                try:
                    camera_status = get_camera_status(self.global_store)

                    if(camera_status == "INACTIVE"):
                        if self.start_engine:
                            update_camera_status(self.global_store,"RUNNING")
                        else:
                            self._stop_event.wait(5)
                    elif(camera_status == "RUNNING"):
                        # Start camera pipeline
                        try:
                            if self.start_engine or get_camera_recovery(self.global_store):
                                self.engine_process = multiprocessing.Process(
                                    target=self.run_engine,
                                    args=(self._stop_event,)
                                )
                                self.start_engine = False
                                self.engine_process.start()
                                time.sleep(2)
                                if not self.engine_process.is_alive():
                                    log.error(f"Process exited early with code {self.engine_process.exitcode}")
                                else:
                                    if get_retries_count(self.global_store) != 0:
                                        update_retries_count(self.global_store, 0)
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
                            self.start_engine = True
                        else:
                            update_camera_status(self.global_store,"INACTIVE")
                            self.start_engine = False
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
                if self.engine_process is not None:
                    self.engine_process.join(timeout=3)
                    if self.engine_process.is_alive():
                        log.warning("Camera engine process force terminating...")
                        self.engine_process.terminate()
                        self.engine_process.join()
                    self.engine_process = None
                log.info("Camera engine process successfully stopped")
            except Exception as e:
                log.error(f"Error while stopping camera pipeline: {e}")
            
            camera_status = get_camera_status(self.global_store)
            if(camera_status == "STOPPED"):
                break
            elif(camera_status == "RESTARTING" or camera_status == "INACTIVE"):
                if camera_status == "RESTARTING":
                    update_camera_status(self.global_store,"RUNNING")
                self._stop_event.clear()

        log.info("Camera stopped running process!!!")

# Update camera_status
def update_camera_status(global_store: GlobalStore, value: str):
    global_store.set("camera_status", value)

# Get camera_status
def get_camera_status(global_store: GlobalStore):
    return global_store.get("camera_status")

# Update camera_recovery
def update_camera_recovery(global_store: GlobalStore, value: bool):
    global_store.set("camera_recovery", value)

# Get camera_status
def get_camera_recovery(global_store: GlobalStore):
    return global_store.get("camera_recovery")

# Update retries_count
def update_retries_count(global_store: GlobalStore, value: int):
    global_store.set("retries_count", value)
    
# Get retries_count
def get_retries_count(global_store: GlobalStore):
    return global_store.get("retries_count")
        