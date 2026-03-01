import os
import json
import asyncio
import logging
import random
import multiprocessing

from lsmy_python_lib.global_store import GlobalStore

from lsmy_python_lib.camera_manager import get_retries_count

from lsmy_python_lib.wifi_config_manager import update_wifi_connect_signal
from lsmy_python_lib.camera_manager import update_camera_status, MAX_RECOVER_TRIES

log = logging.getLogger("ipc")

SOCK = "/run/lsmy/provision.sock"

LAST_TELEMETRY = {
    "temperature": 0.0,
    "humidity": 0.0,
    "no2": 0.0,
    "pm10": 0.0,
    "pm25": 0.0,
}

GLOBAL_STORE = None

async def handle_client(reader, writer):
    try:
        data = await reader.readline()
        if not data:
            return

        req = json.loads(data.decode())
        log.info("IPC RX: %s", req)

        global GLOBAL_STORE

        if req.get("cmd") == "send_telemetry":
            telemetry = {
                "temperature": float(req.get("temperature", 0)),
                "humidity": float(req.get("humidity", 0)),
                "no2": float(req.get("no2", 0)),
                "pm10": float(req.get("pm10", 0)),
                "pm25": float(req.get("pm25", 0)),
            }

            log.info("Telemetry received: %s", telemetry)

            LAST_TELEMETRY.update(telemetry)

            resp = {"status": "ok"}
        elif req.get("cmd") == "request_get_data":
            log.info("Data requested")

            data = {
                "temperature": round(random.uniform(20.0, 35.0), 2),
                "humidity":    round(random.uniform(40.0, 80.0), 2),
                "no2":         round(random.uniform(0.0, 0.5), 4),
                "pm10":        round(random.uniform(10.0, 50.0), 1),
                "pm25":        round(random.uniform(5.0, 25.0), 1),
            }

            resp = {"status": "ok", "data": data}
        elif req.get("cmd") == "connect_wifi_signal":
            role = req.get("role", "hardware")
            status = req.get("status", False)

            update_wifi_connect_signal(GLOBAL_STORE, status)

            log.info("Connect WiFi signal received: role=%s, status=%s", role, status)

            resp = {"status": "ok"}
        elif req.get("cmd") == "update_camera_status":
            status = req.get("status", "INACTIVE")

            if status == "RESTARTING":
                retries_count = get_retries_count(GLOBAL_STORE)
                if retries_count < MAX_RECOVER_TRIES:
                    update_camera_status(GLOBAL_STORE, status)

                data = {
                    "retries_count": retries_count,
                }
                resp = {"status": "ok", "data": data}
            else:
                update_camera_status(GLOBAL_STORE, status)

                log.info("Update camera status received: status=%s", status)

                resp = {"status": "ok"}
        else:
            resp = {"status": "error", "error": "Unknown command"}

        writer.write((json.dumps(resp) + "\n").encode())
        await writer.drain()

    except Exception:
        log.exception("IPC handler error")
    finally:
        writer.close()

async def send_telemetry_ipc(data: dict, timeout=3):
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(SOCK),
        timeout=timeout
    )

    msg = {
        "cmd": "send_telemetry",
        "temperature": data.get("temperature", 0),
        "humidity": data.get("humidity", 0),
        "no2": data.get("no2", 0),
        "pm10": data.get("pm10", 0),
        "pm25": data.get("pm25", 0),
    }

    writer.write((json.dumps(msg) + "\n").encode())
    await writer.drain()

    resp = await reader.readline()
    writer.close()

    return json.loads(resp.decode())

async def send_request_get_data_ipc(timeout=3):
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(SOCK),
        timeout=timeout
    )

    msg = {
        "cmd": "request_get_data",
    }

    writer.write((json.dumps(msg) + "\n").encode())
    await writer.drain()

    resp = await reader.readline()
    writer.close()

    return json.loads(resp.decode())

async def send_connect_wifi_signal_ipc(data: dict, timeout=3):
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(SOCK),
        timeout=timeout
    )

    msg = {
        "cmd": "connect_wifi_signal",
        "role": data.get("role", "hardware"),
        "status": data.get("status", False),
    }

    writer.write((json.dumps(msg) + "\n").encode())
    await writer.drain()

    resp = await reader.readline()
    writer.close()

    return json.loads(resp.decode())

async def send_update_camera_status_signal_ipc(data: dict, timeout=3):
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(SOCK),
        timeout=timeout
    )

    msg = {
        "cmd": "update_camera_status",
        "status": data.get("status", "INACTIVE"),
    }

    writer.write((json.dumps(msg) + "\n").encode())
    await writer.drain()

    resp = await reader.readline()
    writer.close()

    return json.loads(resp.decode())

ipc_loop = None
ipc_stop_event = None

async def ipc_server_task():
    global ipc_stop_event

    if os.path.exists(SOCK):
        os.unlink(SOCK)

    server = await asyncio.start_unix_server(
        handle_client,
        path=SOCK
    )
    os.chmod(SOCK, 0o660)

    log.info("IPC server listening on %s", SOCK)

    async with server:
        await ipc_stop_event.wait()

# -------- IPC Process --------
def start_ipc_process(global_store: GlobalStore, stop_signal, ready_signal):
    log.info("========== STARTING IPC SERVER PROCESS ==========")
    global ipc_loop, ipc_stop_event, GLOBAL_STORE
    GLOBAL_STORE = global_store

    ipc_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ipc_loop)

    ipc_stop_event = asyncio.Event()

    async def watch_stop_signal():
        while not stop_signal.is_set():
            await asyncio.sleep(0.5) 
        
        log.info("IPC Process received stop signal from Main Process")
        ipc_stop_event.set()

    try:
        ipc_loop.create_task(watch_stop_signal())
        ipc_loop.run_until_complete(ipc_server_task())
        ready_signal.set()
    finally:
        ipc_loop.close()
        unlink_ipc_socket()
        ready_signal.clear()

def stop_ipc_process(ipc_process, ipc_stop_signal):
    log.info("========== STOPPING IPC SERVER PROCESS ==========")

    if ipc_process.is_alive():
        ipc_stop_signal.set()

def unlink_ipc_socket():
    if os.path.exists(SOCK):
        os.unlink(SOCK)
        log.info("IPC Socket removed.")

