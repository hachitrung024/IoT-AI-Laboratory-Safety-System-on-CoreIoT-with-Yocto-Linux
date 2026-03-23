#!/usr/bin/python3
# =============================================================================
#  LSMY System Entrypoint
# -----------------------------------------------------------------------------
#  File        : run_lsmy.py
#  Role        : Application entrypoint
#
#  Responsibilities:
#   - Initialize runtime environment
#   - Bootstrap LSMY application
#   - Start main application loop
#   - Handle top-level exceptions & exit codes
#
#  IMPORTANT:
#   - No business logic here
#   - Keep this file minimal and clean
#   - All logic must live inside lsmy-python-app package
# =============================================================================

import os
import sys
import ctypes
import logging
from logging.handlers import RotatingFileHandler

# -------------------------
# Logging Configuration
# -------------------------
LOG_DIR = "/data/logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler("/data/logs/run-lsmy.log", maxBytes=1*1024*1024, backupCount=2),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger("run-lsmy")

############################################
# PYTHON APPLICATION FOR LSMY              #
# - LSMY Application                       #
# - All logic must live inside here        #
############################################
from lsmy_app.app import LsmyApplication


# -------------------------
# Main entrypoint
# -------------------------
def main() -> int:
    """
    Main process entrypoint.

    Returns:
        int: Unix exit code
             0   -> Normal exit
             !=0 -> Error
    """

    try:
        app = LsmyApplication()
        app.start()

    except KeyboardInterrupt:
        log.warning("Shutdown requested by user with KeyboardInterrupt)")
        return 0

    except Exception as exc:
        log.exception("Fatal error during application startup")
        return 1
    finally:
        if app:
            app.stop()

    log.info("LSMY System exited normally")
    return 0


# -------------------------
# Process bootstrap
# -------------------------
if __name__ == "__main__":
    sys.exit(main())
