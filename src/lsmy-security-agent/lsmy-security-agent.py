#!/usr/bin/env python3

import hashlib
import time
import subprocess
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

BASELINE_FILE = "/etc/security/baseline.db"
WHITELIST_FILE = "/etc/security/process_whitelist.txt"
LOG_FILE = "/data/logs/security-agent.log"

LOG_DIR = "/data/logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1*1024*1024, backupCount=2),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger("security-agent")

CHECK_INTERVAL = 30

# Hash a file
def sha256_file(path):
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
    except Exception as e:
        log.error(f"Error hashing {path}: {e}")
        return None

# Load integrity baseline
def load_baseline(filepath):
    baseline = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    expected_hash, path = parts
                    baseline[path] = expected_hash
    except FileNotFoundError:
        log.error(f"Baseline file not found at {filepath}")
    return baseline

# Check file integrity
def check_files(baseline):
    failed_files = []
    missing_files = []
    
    for path, expected_hash in baseline.items():
        if not os.path.exists(path):
            log.warning(f"Missing file: {path}")
            missing_files.append(path)
            continue

        current_hash = sha256_file(path)

        if current_hash != expected_hash:
            log.warning(f"File modified: {path}")
            failed_files.append(path)

    if not failed_files and not missing_files:
        log.info("Integrity check passed: All files are valid.")
        return True
    else:
        log.error(f"Integrity check FAILED: {len(failed_files)} modified, {len(missing_files)} missing.")
        return False

# Load whitelist process
def load_whitelist():
    wl = set()
    with open(WHITELIST_FILE) as f:
        for line in f:
            wl.add(line.strip())
    return wl

def check_process(whitelist):
    try:
        ps = subprocess.check_output(["ps", "-eo", "comm"]).decode().splitlines()
    except:
        return False

    for p in ps:
        p = p.strip()
        if p and p not in whitelist:
            log.warning(f"Unknown process: {p}")
            return False

    return True

LSMY_SERVICE = "run-lsmy.service"

# Action when tampering detected
def trigger_response():
    log.warning("Tamper detected -> Rebooting system....")
    run_cmd_with_retry(
            ["systemctl", "stop", LSMY_SERVICE]
        )
    # os.system("reboot")

# -- Main --
def main():
    global CHECK_INTERVAL
    log.info("Security agent service started")

    baseline = load_baseline(BASELINE_FILE)
    # whitelist = load_whitelist()

    log.info(f"Monitoring {len(baseline)} files from baseline.")

    while True:
        start_time = time.time()

        ok_files = check_files(baseline)
        # ok_proc = check_process(whitelist)
        ok_proc = True

        elapsed = time.time() - start_time
        if elapsed > CHECK_INTERVAL:
            log.warning(f"Scan took {elapsed:.2f}s, which is longer than {CHECK_INTERVAL}s!")
            CHECK_INTERVAL = elapsed + 60*3
            log.info(f"Updated CHECK_INTERVAL: {CHECK_INTERVAL}")

        if not ok_files or not ok_proc:
            trigger_response()
            break

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()