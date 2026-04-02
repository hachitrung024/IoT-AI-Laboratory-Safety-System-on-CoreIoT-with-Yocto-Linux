#!/usr/bin/env python3

import os
import sys
import time
import socket
import struct
import hashlib
import logging
import threading
import subprocess
from logging.handlers import RotatingFileHandler

# ====== COMMAND RUNNER LIBRARY ======
from lsmy_python_lib.command_runner import run_cmd, run_cmd_with_retry

LSMY_SERVICE = "run-lsmy.service"

SEC_SOCKET_PATH = "/run/lsmy/security.sock"
MAINTENANCE_TOKEN = "LSMY_PAUSE_SECRET_99"
MAINTENANCE_MODE = False

BASELINE_FILE = "/etc/security/baseline.db"
WHITELIST_FILE = "/etc/security/whitelist.txt"
GOLD_BACKUP = "/etc/security/gold_backup.tar.gz"

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

    if not failed_files:
        if not missing_files:
            log.info("Integrity check passed: All files are valid.")
        else:
            log.info(f"Integrity check passed: Have {len(missing_files)} missing.")
        return True, []
    else:
        log.error(f"Integrity check failed: {len(failed_files)} modified, {len(missing_files)} missing.")
        return False, failed_files

# Load whitelist of packages
def load_whitelist():
    wl = set()
    try:
        if not os.path.exists(WHITELIST_FILE):
            log.error(f"Whitelist file not found at {WHITELIST_FILE}")
            return wl
        with open(WHITELIST_FILE, 'r') as f:
            for line in f:
                pkg = line.strip()
                if pkg and not pkg.startswith('#'):
                    wl.add(pkg)
    except Exception as e:
        log.error(f"Error loading whitelist: {e}")
    return wl

def check_packages(whitelist):
    intrusion_packages = []
    try:
        result = subprocess.run(["opkg", "list-installed"], capture_output=True, text=True, check=True)
        
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            
            pkg_name = line.split(' - ')[0].strip()

            if pkg_name not in whitelist:
                if pkg_name.startswith('kernel') or pkg_name.startswith('libgcc') or \
                   pkg_name in ['libc6', 'opkg', 'base-files', 'busybox', 'lsmy-security-agent']:
                    continue
                
                log.warning(f"Unauthorized package detected: {pkg_name}")
                intrusion_packages.append(pkg_name)

        if intrusion_packages:
            log.error(f"Package integrity failed: {len(intrusion_packages)} unknown packages.")
            return False, intrusion_packages

    except subprocess.CalledProcessError as e:
        log.error(f"Could not execute opkg: {e}")
        return True, [] 
    except Exception as e:
        log.error(f"Error during package check: {e}")
        return True, []

    log.info("Package check passed: No unauthorized software found.")
    return True, []

# Action when tampering detected
def trigger_response(files_to_restore, packages_to_uninstall):
    log.warning("Detected tampering. Stopping system services...")
    run_cmd_with_retry(
            ["systemctl", "stop", LSMY_SERVICE]
        )
    
    log.warning("!!! TAMPERING DETECTED !!!")
    
    if files_to_restore != []:
        log.info("Restoring system from backup...")
        log.info(f"Selective restore: {files_to_restore}")
        try:
            cleaned_files = [f.lstrip('/') for f in files_to_restore]
            cmd = ["tar", "-xzvf", GOLD_BACKUP, "-C", "/", "--overwrite"] + cleaned_files
            if not MAINTENANCE_MODE:
                subprocess.run(cmd, check=True)
                log.info("Restore successful!")
        except subprocess.CalledProcessError as e:
            log.error(f"Restore failed: {e}")

    if packages_to_uninstall != []:
        log.info("Uninstalling introsion packages...")
        log.info(f"Selective uninstall: {packages_to_uninstall}")
        try:
            # cmd = [
            #     "opkg", "remove", 
            #     "--force-removal-of-dependent-packages", 
            #     "--autoremove"
            # ] + packages_to_uninstall
            
            # if not MAINTENANCE_MODE:
                # log.info(f"Running command: {' '.join(cmd)}")
                
                # result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # log.info("Uninstall successful!")
                # log.debug(f"Opkg output: {result.stdout}")
            pass

        except subprocess.CalledProcessError as e:
            log.error(f"Uninstall failed! Exit code: {e.returncode}")
            log.error(f"Opkg Error: {e.stderr}")
    
    log.warning("System rebooting in 5 seconds...")
    time.sleep(5) 
    # run_cmd(["sudo", "reboot"], check=False)

def socket_control_thread():
    global MAINTENANCE_MODE
    
    log.info("========== STARTING SOCKET CONTROL THREAD ==========")

    socket_dir = os.path.dirname(SEC_SOCKET_PATH)
    if not os.path.exists(socket_dir):
        try:
            log.info(f"Creating directory: {socket_dir}")
            os.makedirs(socket_dir, mode=0o755, exist_ok=True)
        except Exception as e:
            log.error(f"Could not create socket directory: {e}")
            return
        
    if os.path.exists(SEC_SOCKET_PATH):
        os.remove(SEC_SOCKET_PATH)
    
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SEC_SOCKET_PATH)
    os.chmod(SEC_SOCKET_PATH, 0o600)
    server.listen(1)

    while True:
        conn, _ = server.accept()
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
            pid, uid, gid = struct.unpack('3i', creds)
            
            if uid != 0:
                conn.sendall(b"Access Denied: Root only\n")
                conn.close()
                continue

            data = conn.recv(1024).decode().strip()
            if data == f"START_MAINTENANCE:{MAINTENANCE_TOKEN}":
                MAINTENANCE_MODE = True
                log.warning("!!! SECURITY PAUSED VIA SOCKET !!!")
                conn.sendall(b"ACK:PAUSED\n")
            elif data == f"STOP_MAINTENANCE:{MAINTENANCE_TOKEN}":
                MAINTENANCE_MODE = False
                log.info("--- SECURITY RESUMED VIA SOCKET ---")
                conn.sendall(b"ACK:RESUMED\n")
            else:
                log.warning(f"Invalid command: {data}")
                conn.sendall(b"Invalid command\n")
            
        except Exception as e:
            log.error(f"Socket error: {e}")
        finally:
            conn.close()

def is_early_boot():
    with open("/proc/uptime") as f:
        uptime = float(f.read().split()[0])
    return uptime < 120

INIT_FLAG = "/etc/security/baseline_initialized"

def should_refresh_baseline():
    if not os.path.exists(BASELINE_FILE):
        log.warning("Baseline missing, rebuild allowed")
        return True

    baseline = load_baseline(BASELINE_FILE)
    ok, _ = check_files(baseline)

    if ok:
        log.info("System clean, do not need to refresh baseline")
        return False
    else:
        if not os.path.exists(INIT_FLAG):
            log.warning("System modified, need to rebuild baseline!")
            return True
        else:
            log.info("System modified, but baseline already initialized")
            return False

def refresh_baseline_once_after_boot():
    try:
        if is_early_boot():
            if should_refresh_baseline():
                with open(INIT_FLAG, "w") as f:
                    f.write("initialized\n")

                # Make the file read-only
                os.chmod(INIT_FLAG, 0o400)
                try:
                    subprocess.run(["chattr", "+i", INIT_FLAG], check=True)

                    if os.path.exists(BASELINE_FILE):
                        subprocess.run(["chattr", "-i", BASELINE_FILE], check=True)

                    env = os.environ.copy()
                    env["ROOTFS"] = ""
                    subprocess.run(["/usr/bin/gen_baseline.sh"], check=True)

                    if os.path.exists(BASELINE_FILE):
                        subprocess.run(["chattr", "+i", BASELINE_FILE], check=False)

                    log.info("Boot baseline refreshed successfully.")
                except Exception:
                    log.error("Boot baseline refresh failed.")

    except Exception as e:
        log.exception(f"Boot baseline refresh failed: {e}") 

# -- Main --
def main():
    global CHECK_INTERVAL
    log.info("========== STARTING SECURITY AGENT SERVICES ==========")

    t = threading.Thread(target=socket_control_thread, daemon=True)
    t.start()

    refresh_baseline_once_after_boot()

    baseline = load_baseline(BASELINE_FILE)
    whitelist = load_whitelist()

    log.info(f"Monitoring {len(baseline)} files from baseline.")
    log.info(f"Monitoring {len(whitelist)} packages from whitelist.")

    while True:
        if MAINTENANCE_MODE:
            log.info("Maintenance mode active... sleeping.")
            time.sleep(10)
            continue

        start_time = time.time()

        ok_files, failed_files = check_files(baseline)
        ok_proc, introsion_packages = check_packages(whitelist)

        elapsed = time.time() - start_time
        if elapsed > CHECK_INTERVAL:
            log.warning(f"Scan took {elapsed:.2f}s, which is longer than {CHECK_INTERVAL}s check interval!")
            CHECK_INTERVAL = elapsed + 60*3
            log.info(f"Updated check interval: {CHECK_INTERVAL}")

        if not ok_files or not ok_proc:
            trigger_response(failed_files, introsion_packages)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()