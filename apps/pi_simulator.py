"""
pi_simulator.py — Ward Watcher Raspberry Pi Debug Simulator.

Impersonates one or more Raspberry Pi devices by:
  - Pinging POST /api/ping (with room_id) every 2 seconds per room,
    so the dashboard shows each room as 'Connected'.
  - Providing an interactive menu or --auto mode to fire test alerts.

Usage
-----
    # Single room (default room 1)
    python pi_simulator.py

    # Multiple rooms — each gets its own heartbeat thread
    python pi_simulator.py --rooms 1,2,3

    # Auto-mode: random alert every 10 s from a random room
    python pi_simulator.py --rooms 1,2,3 --auto

    # Faster auto with fixed room and custom interval
    python pi_simulator.py --rooms 2 --auto --interval 5

    # Point at a remote server
    python pi_simulator.py --rooms 1,2 --url http://192.168.1.10:5000
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Alert type config ──────────────────────────────────────────────────

ALERT_TYPES: list[str] = [
    "fire",
    "unauthorized_presence",
    "unauthorized_touch",
]

ALERT_LABELS: dict[str, str] = {
    "fire":                  "Fire",
    "unauthorized_presence": "Unauthorized Presence",
    "unauthorized_touch":    "Unauthorized Touch",
}

HEARTBEAT_INTERVAL = 2.0  # seconds between pings per room


# ── Simulator ──────────────────────────────────────────────────────────

class PiSimulator:
    """
    Simulates one or more Raspberry Pi devices.

    Each room runs its own background heartbeat thread so rooms are
    independently tracked on the dashboard.
    """

    def __init__(self, base_url: str, rooms: list[int]) -> None:
        self.base_url  = base_url.rstrip("/")
        self.rooms     = rooms
        self._stop_evt = threading.Event()

    # ── Heartbeat ──────────────────────────────────────────────────────

    def start_heartbeats(self) -> None:
        """Spawn one heartbeat thread per room. Returns immediately."""
        for room_id in self.rooms:
            t = threading.Thread(
                target=self._heartbeat_loop,
                args=(room_id,),
                daemon=True,
                name=f"hb-room-{room_id}",
            )
            t.start()

    def _heartbeat_loop(self, room_id: int) -> None:
        first_success = False
        while not self._stop_evt.is_set():
            try:
                requests.post(
                    f"{self.base_url}/api/ping",
                    json={"room_id": room_id},
                    timeout=2,
                )
                if not first_success:
                    first_success = True
                    print(f"[HB] Room {room_id} — connected to server.")
            except requests.exceptions.ConnectionError:
                print(f"[HB] Room {room_id} — server unreachable, retrying…")
            except Exception as exc:
                print(f"[HB] Room {room_id} — error: {exc}")

            self._stop_evt.wait(HEARTBEAT_INTERVAL)

    def stop(self) -> None:
        self._stop_evt.set()

    # ── Alerts ─────────────────────────────────────────────────────────

    def send_alert(self, alert_type: str, room_id: int) -> dict:
        """POST an alert to the server and return the response body."""
        payload = {
            "alert_type": alert_type,
            "room_id":    room_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        resp = requests.post(f"{self.base_url}/alert", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def _fire(self, alert_type: str, room_id: int) -> None:
        try:
            result   = self.send_alert(alert_type, room_id)
            alert_id = result.get("alert_id", "?")
            label    = ALERT_LABELS.get(alert_type, alert_type)
            print(f"[ALERT #{alert_id}] {label} | Room {room_id}")
        except requests.exceptions.ConnectionError:
            print("[ALERT] Failed — server not reachable.")
        except requests.exceptions.HTTPError as exc:
            print(f"[ALERT] Server rejected: {exc.response.text}")
        except Exception as exc:
            print(f"[ALERT] Error: {exc}")

    def fire_random(self) -> None:
        """Send a random alert type from a random simulated room."""
        self._fire(
            alert_type=random.choice(ALERT_TYPES),
            room_id=random.choice(self.rooms),
        )

    # ── Run modes ──────────────────────────────────────────────────────

    def run_interactive(self) -> None:
        """Interactive command loop — blocks until the user quits."""
        room_list = ", ".join(str(r) for r in self.rooms)
        print(f"""
╔══════════════════════════════════════════════════╗
║   Ward Watcher  ·  Pi Simulator                  ║
╠══════════════════════════════════════════════════╣
║  1  Send Fire alert                              ║
║  2  Send Unauthorized Presence alert             ║
║  3  Send Unauthorized Touch alert                ║
║  r  Send a random alert from a random room       ║
║  q  Quit                                         ║
╚══════════════════════════════════════════════════╝
  Server  : {self.base_url}
  Rooms   : {room_list}
  Heartbeat pinging every 2 s per room.

  Tip: append a room number to target a specific room.
  Examples:  1       → fire alert to room {self.rooms[0]}
             1 2     → fire alert to room 2
             r 3     → random alert to room 3
""")

        choices = {
            "1": "fire",
            "2": "unauthorized_presence",
            "3": "unauthorized_touch",
        }

        while True:
            try:
                raw = input("Command > ").strip().lower().split()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            cmd      = raw[0]
            room_arg = raw[1] if len(raw) > 1 else None

            # Resolve room_id: explicit arg → random from pool
            try:
                room_id = int(room_arg) if room_arg else self.rooms[0]
            except ValueError:
                print(f"  Room ID must be an integer.")
                continue

            if cmd in choices:
                self._fire(choices[cmd], room_id)
            elif cmd == "r":
                self._fire(random.choice(ALERT_TYPES), room_id)
            elif cmd == "q":
                break
            else:
                print("  Unknown command. Use 1 / 2 / 3 / r / q.")

        print("[SIM] Exiting.")

    def run_auto(self, interval: float) -> None:
        """Auto-mode: fire a random alert every `interval` seconds."""
        room_list = ", ".join(str(r) for r in self.rooms)
        print(f"[AUTO] Rooms {room_list} — random alert every {interval}s. Ctrl+C to stop.\n")
        try:
            while True:
                self.fire_random()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[AUTO] Stopped.")


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ward Watcher Raspberry Pi Debug Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Base URL of the Ward Watcher server",
    )

    room_group = p.add_mutually_exclusive_group()
    room_group.add_argument(
        "--room",
        type=int,
        default=None,
        metavar="N",
        help="Single room ID to simulate (shorthand for --rooms N)",
    )
    room_group.add_argument(
        "--rooms",
        type=str,
        default=None,
        metavar="1,2,3",
        help="Comma-separated list of room IDs to simulate simultaneously",
    )

    p.add_argument(
        "--auto",
        action="store_true",
        help="Auto-send random alerts instead of showing the interactive menu",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Seconds between alerts in auto mode",
    )
    return p.parse_args()


def _resolve_rooms(args: argparse.Namespace) -> list[int]:
    if args.rooms:
        try:
            return [int(r.strip()) for r in args.rooms.split(",") if r.strip()]
        except ValueError:
            raise SystemExit("--rooms must be a comma-separated list of integers, e.g. 1,2,3")
    if args.room is not None:
        return [args.room]
    return [1]  # default


def main() -> None:
    args  = _parse_args()
    rooms = _resolve_rooms(args)
    sim   = PiSimulator(base_url=args.url, rooms=rooms)

    sim.start_heartbeats()
    time.sleep(0.4)  # let first pings land before printing the menu

    if args.auto:
        sim.run_auto(interval=args.interval)
    else:
        sim.run_interactive()

    sim.stop()


if __name__ == "__main__":
    main()
