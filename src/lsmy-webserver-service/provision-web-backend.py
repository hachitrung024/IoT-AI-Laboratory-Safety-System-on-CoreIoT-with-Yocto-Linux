#!/usr/bin/python3
import os
import re
import sys
import asyncio
import json
import subprocess
import logging
import websockets
import gpiod

# ====== IPC LIBRARY ======
from lsmy_python_lib.ipc import send_connect_wifi_signal_ipc, send_request_get_data_ipc, LAST_TELEMETRY

# ====== WIFI CONFIG LIBRARY ======
from lsmy_python_lib.wifi_config_manager import configure_wifi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("provision-webserver-backend")

clients = set()
WS_PORT = 8765

# WebSocket handler
async def handle(ws):
    clients.add(ws)
    log.info("Client connected (%d)", len(clients))

    try:
        async for msg in ws:
            log.info("RX: %s", msg)
            data = json.loads(msg)

            # ================= SETTINGS =================
            if data.get("page") == "setting":
                action = data.get("action")
                ssid = data["value"]["ssid"]
                password = data["value"]["password"]

                clean_ssid = ssid.strip() if ssid else ""
                clean_pw = password.strip() if password else ""

                if not clean_ssid:
                    await ws.send(json.dumps({
                        "status": "error",
                        "msg": "SSID is required"
                    }))
                else:
                    if action == "connectBtn":
                        # Update the wifi config signal
                        data = {
                            "role": "backend",
                            "status": True 
                        }
                        await send_connect_wifi_signal_ipc(data)
                        await ws.send(json.dumps({
                        "status": "ok",
                        "msg": "WiFi connect signal successfully"
                        }))
                    elif action == "saveBtn":
                        # Configure WiFi
                        data = {
                            "role": "backend",
                            "status": False 
                        }
                        await send_connect_wifi_signal_ipc(data)
                        configure_wifi(clean_ssid, clean_pw)

                        await ws.send(json.dumps({
                        "status": "ok",
                        "msg": "WiFi configured successfully"
                        }))
            # ================= DEVICE / RELAY =================
            elif data.get("page") == "device":
                log.info("Relay command: %s", data["value"])

                value = data.get("value", {})
                # Here is relay control logic
                try:
                    gpio = int(value.get("gpio"))
                    status = value.get("status")
                    name = value.get("name", "Unknown")

                    if status not in ("ON", "OFF"):
                        raise ValueError("Invalid relay status")

                    set_gpio(gpio, status == "ON")

                    await ws.send(json.dumps({
                        "status": "ok",
                        "msg": f"Relay executed {name} turned {status}",
                        "gpio": gpio,
                        "state": status
                    }))

                except Exception as e:
                    log.error("Relay control failed: %s", e)

                    await ws.send(json.dumps({
                        "status": "error",
                        "msg": str(e)
                    }))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.remove(ws)
        log.info("Client disconnected (%d)", len(clients))

# WebSocket server task
async def ws_server_task():
    async with websockets.serve(handle, "0.0.0.0", WS_PORT):
        log.info("WiFi provision WS running on %d", WS_PORT)
        await asyncio.Future()

# Telemetry broadcast task            
async def telemetry_task():
    while True:
        if clients:
            sensor = await read_sensors()
            payload = {
                "clients": len(clients),
                **sensor
            }

            msg = json.dumps(payload)

            await asyncio.gather(
                *[ws.send(msg) for ws in clients],
                return_exceptions=True
            )

            log.info("TX telemetry: %s", msg)

        await asyncio.sleep(5)  
    
async def read_sensors():
    try:
        resq = await send_request_get_data_ipc()
        
        if resq.get("status") == "ok":
            data = resq["data"]
            return {
                "temperature": data["temperature"],
                "humidity":    data["humidity"],
                "no2":         data["no2"],
                "pm10":        data["pm10"],
                "pm25":        data["pm25"],
            }
        else:
            log.error("IPC Server returned error: %s", resq.get("error"))
            return None
    except Exception as e:
        log.error("Failed to read sensors via IPC: %s", e)
        return None

GPIO_CHIP = '/dev/gpiochip4'
GPIO_LINE = 16

def set_gpio(gpio_num: int, on: bool):
    try:
        with gpiod.request_lines(
            GPIO_CHIP,
            consumer="lsmy-provision",
            config={
                gpio_num: gpiod.LineSettings(
                    direction=gpiod.Direction.OUTPUT,
                    output_value=gpiod.Value.ACTIVE if on else gpiod.Value.INACTIVE
                )
            }
        ) as request:
            log.info("GPIO %d set to %s", gpio_num, "ON" if on else "OFF")
    except Exception as e:
        log.error("GPIO %d control failed: %s", gpio_num, e)
        raise

def shutdown_provision():
    subprocess.run(["systemctl", "stop", "provision-web-backend"])


async def main():
    await asyncio.gather(
        ws_server_task(),
        telemetry_task(),
    )

asyncio.run(main())