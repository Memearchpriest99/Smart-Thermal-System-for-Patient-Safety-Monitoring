# Ward Watcher — Hospital Surveillance Dashboard

A real-time browser-based surveillance dashboard for hospital room monitoring. Each room's Raspberry Pi posts alerts and heartbeat pings to the server; the dashboard updates instantly via WebSocket, plays an audible alarm until the alert is dismissed, and shows a live connection status card for every room.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup (Development)](#setup-development)
- [Running in Development](#running-in-development)
- [Building the Standalone Windows Package](#building-the-standalone-windows-package)
- [Hospital Deployment Guide](#hospital-deployment-guide)
- [Alert Types](#alert-types)
- [API Endpoints](#api-endpoints)
- [SocketIO Events](#socketio-events)
- [Database Schema](#database-schema)
- [Dashboard](#dashboard-indexhtml)
- [Database Manager](#database-manager-dbhtml)
- [Pi Simulator](#pi-simulator-pi_simulatorpy)
- [File Reference](#file-reference)

---

## Project Structure

```
HospitalSurveillance/
├── app.py                  # Flask server — all routes and SocketIO events
├── database.py             # SQLite layer — all read/write operations
├── pi_simulator.py         # Debug tool — impersonates one or more Raspberry Pis
├── pi_client.py            # Runs on the actual Raspberry Pi
│
├── ward_watcher.spec       # PyInstaller build spec
├── build_windows.bat       # One-click build script (Windows)
│
├── templates/
│   ├── index.html          # Live surveillance dashboard (markup only)
│   └── db.html             # Database manager UI (markup only)
│
└── static/
    ├── css/
    │   ├── index.css       # Dashboard styles
    │   └── db.css          # Database manager styles
    └── js/
        ├── index.js        # Dashboard logic (room tracking, sound, alerts)
        ├── db.js           # Database manager logic (CRUD, modal, tabs)
        └── socket.io.min.js  # Socket.IO client (bundled — no internet needed)
```

> `ward_watcher.db` is created automatically next to `app.py` (or next to `ward_watcher.exe` when packaged) on first run.

---

## Setup (Development)

**Requirements:** Python 3.8+

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux / Mac

pip install flask flask-socketio
```

No other dependencies. SQLite is part of the Python standard library. Socket.IO JS is bundled in `static/js/` and requires no internet connection.

---

## Running in Development

```bash
# Activate venv first
venv\Scripts\activate

python app.py
```

Open `http://localhost:5000` in a browser. The header shows **"Connecting…"** and the Room Connections panel shows **"Waiting for rooms to connect…"** until a Pi (or the simulator) starts pinging.

To simulate the Pi in a second terminal:

```bash
python pi_simulator.py --rooms 1,2,3
```

---

## Building the Standalone Windows Package

This produces a self-contained folder that runs on any Windows PC — no Python, no internet, no installation required.

### One-click build

```bat
build_windows.bat
```

The script activates the venv, installs PyInstaller if needed, and produces:

```
dist\ward_watcher\
    ward_watcher.exe
    _internal\         ← Python runtime + all dependencies
        templates\
        static\
        ...
```

### Manual build

```bash
venv\Scripts\activate
pyinstaller ward_watcher.spec --clean --noconfirm
```

### Output size

~25 MB total. Copy the entire `dist\ward_watcher\` folder to a USB stick or hospital PC.

---

## Hospital Deployment Guide

Follow these steps in order when you arrive on site.

---

### Step 1 — Set up the network

All devices must be on the **same local network**. Two options:

| Option | How |
|---|---|
| **Phone hotspot** (recommended for portability) | Turn on hotspot on your phone. Connect both the PC and the Pi to it. |
| **Hospital ethernet / Wi-Fi** | Connect both devices to the same hospital network. Confirm with IT that the port 5000 is not blocked between them. |

---

### Step 2 — Find the server PC's IP address

On the PC running Ward Watcher, open PowerShell or Command Prompt:

```powershell
ipconfig
```

Look for **IPv4 Address** under the active adapter (e.g. `10.243.12.176`). This is `SERVER_IP`. Write it down.

---

### Step 3 — Start the Ward Watcher server

**If using the packaged exe (hospital PC or own laptop without Python):**

1. Copy `dist\ward_watcher\` to the PC (USB stick or direct copy).
2. Double-click `ward_watcher.exe`.
3. A console window opens. Wait for: `[SERVER] Ward Watcher starting → http://localhost:5000`
4. A browser tab opens automatically at `http://localhost:5000`.

**If running from source (own laptop with Python):**

```bash
cd HospitalSurveillance
venv\Scripts\activate
python app.py
```

Then open `http://localhost:5000` in your browser.

---

### Step 4 — Allow port 5000 through the Windows firewall

The Pi needs to reach port 5000 on the PC. Run this once in an **Administrator** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Ward Watcher" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

To verify the server is reachable from the Pi's side, you can test from the Pi:

```bash
curl http://SERVER_IP:5000
```

If you get HTML back, the connection is open.

---

### Step 5 — Configure the Pi

SSH into the Pi or use VNC. Edit `pi_client.py` and update the `SERVER` line:

```python
SERVER  = "http://10.243.12.176:5000"   # ← replace with your actual SERVER_IP
ROOM_ID = 1                              # ← set the room number for this Pi
```

**Finding the Pi's IP** (if you don't have it): check your phone hotspot's connected devices list, or from the Pi run `hostname -I`.

---

### Step 6 — Start the Pi client

On the Pi:

```bash
python3 ~/pi_client.py
```

You should see:

```
Heartbeat running -> http://10.243.12.176:5000  (Room 1)
1=Fire  2=Unauthorized Presence  3=Unauthorized Touch  q=Quit
Alert >
```

On the dashboard, a **Room 1** card appears with a green pulsing dot within 3 seconds.

---

### Step 7 — Verify everything is working

| Check | Expected result |
|---|---|
| Dashboard header dot | Green — **Server Online** |
| Room Connections panel | Room card(s) visible, green border, pulsing dot |
| Pi console | `Heartbeat running →` (no error messages) |
| Send a test alert from Pi | Type `1` and press Enter — a red Fire card appears on the dashboard |
| Dismiss the alert | Click **Dismiss** — card disappears on all open tabs |
| Pull the Pi's network cable / stop pi_client.py | Room card turns red within 3 seconds |

---

### Step 8 — Operating the dashboard

| Element | What it means |
|---|---|
| Green header dot — **Server Online** | Browser is connected to the Flask server via WebSocket |
| Red header dot — **Server Offline** | Flask server is unreachable (check that `ward_watcher.exe` is still running) |
| Room card — green border, pulsing dot | Pi for that room is sending heartbeats normally |
| Room card — red border, pulsing border | No heartbeat received from that room for >3 seconds (Pi offline or network issue) |
| Alert card appears | Sensor triggered; beep starts if sound is enabled |
| **Dismiss** button | Marks alert resolved; stops beep if no other alerts remain |
| **🔇 Sound** button | Click once to enable alarm beep; click again to mute |
| **Database →** button | Opens the alert history / audit log interface |

**Sound is off by default** (browser autoplay policy). Click the Sound button once at the start of each session to enable it.

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| Room card never appears | Pi can't reach the server. Check both are on the same network; check `SERVER_IP` in `pi_client.py`; check the firewall rule (Step 4). |
| Room card immediately turns red | Pi is stopping/crashing. Check Pi console for errors. |
| Browser shows "Connecting…" indefinitely | Flask server is not running or browser is pointing at the wrong IP/port. |
| `ward_watcher.exe` console shows `[CONNECT]` but no room card | Pi is pinging but WebSocket isn't reaching the browser. Try refreshing the tab. |
| Alert appears but no sound | Click the **Sound** button to enable it (must be done once per session). |
| Database file missing after update | `ward_watcher.db` lives next to `ward_watcher.exe`. Do not delete the whole folder between updates — only replace the exe. |

---

## Alert Types

| `alert_type`             | Display Name           | Color  |
|--------------------------|------------------------|--------|
| `fire`                   | Fire Detected          | Red    |
| `unauthorized_presence`  | Unauthorized Presence  | Orange |
| `unauthorized_touch`     | Unauthorized Touch     | Purple |

---

## API Endpoints

### `POST /alert`

Receives an alert from a room's Pi or sensor. Stores it in the database and broadcasts it to all connected dashboard clients.

**Request body:**
```json
{
  "alert_type": "fire",
  "room_id": 1,
  "timestamp": "2026-06-28T09:15:00+00:00"
}
```

- `timestamp` is optional — the server uses current UTC time if omitted.
- `alert_id` is always assigned by the server (SQLite `AUTOINCREMENT`).

**Response:**
```json
{ "status": "ok", "alert_id": 42 }
```

---

### `POST /api/ping`

Receives a per-room liveness heartbeat from a Pi. Relays it to all dashboard clients as a `room_heartbeat` SocketIO event.

Each room's Pi calls this every ~2 seconds. If pings stop for a room, that room's card turns red after 3 seconds.

**Request body:**
```json
{ "room_id": 1 }
```

---

### `GET /`, `GET /db`

Serve the live dashboard and database manager respectively.

---

### `GET /api/alerts`

Returns all alert records as a JSON array, newest first.

---

### `POST /api/alerts`

Creates an alert record via the DB manager UI. Does **not** broadcast a `new_alert` event.

---

### `PUT /api/alerts/<alert_id>`

Updates an existing alert record. Accepts any subset of: `alert_type`, `room_id`, `timestamp`, `dismissed`, `dismissed_at`.

---

### `DELETE /api/alerts/<alert_id>`

Deletes an alert record and all associated dismiss log entries.

---

### `GET /api/dismiss-log`

Returns all dismiss log entries, newest first.

---

### `DELETE /api/dismiss-log/<log_id>`

Deletes a single dismiss log entry.

---

## SocketIO Events

### Server → Client

| Event             | Payload                                         | Description |
|-------------------|-------------------------------------------------|-------------|
| `room_heartbeat`  | `{ "room_id": int, "server_time": float }`     | Relayed from a Pi calling `POST /api/ping`. Resets that room's 3-second timeout. |
| `new_alert`       | `{ alert_id, alert_type, room_id, timestamp }` | Broadcast when `POST /alert` is received. All open tabs show the card immediately. |
| `alert_dismissed` | `{ "alert_id": int }`                          | Broadcast on dismissal. All other open tabs remove that card. |

### Client → Server

| Event           | Payload               | Description |
|-----------------|-----------------------|-------------|
| `dismiss_alert` | `{ "alert_id": int }` | Sent when the user clicks Dismiss. Server marks the alert dismissed and broadcasts `alert_dismissed`. |

---

## Database Schema

Two tables stored in `ward_watcher.db`.

### `alerts`

| Column         | Type    | Description |
|----------------|---------|-------------|
| `alert_id`     | INTEGER | Primary key, auto-assigned by the server. |
| `alert_type`   | TEXT    | One of the three alert type strings. |
| `room_id`      | INTEGER | Identifier of the room the alert originated from. |
| `timestamp`    | TEXT    | ISO 8601 datetime string (from the Pi or server). |
| `dismissed`    | INTEGER | `0` = active, `1` = dismissed. |
| `dismissed_at` | TEXT    | ISO 8601 datetime of dismissal, or `NULL` if still active. |

### `dismiss_log`

Append-only audit trail — a row is added every time an alert is dismissed.

| Column         | Type    | Description |
|----------------|---------|-------------|
| `log_id`       | INTEGER | Primary key. |
| `alert_id`     | INTEGER | The dismissed alert's ID. |
| `alert_type`   | TEXT    | Copied from the alert at dismissal time. |
| `room_id`      | INTEGER | Copied from the alert at dismissal time. |
| `dismissed_at` | TEXT    | ISO 8601 datetime of the dismiss action. |

---

## Dashboard (`index.html`)

The main real-time view at `http://localhost:5000`.

### Header

| Element | Description |
|---------|-------------|
| **Server dot + label** | Socket.IO connectivity indicator. Green = server reachable, red = server offline. |
| **Sound** button | Toggles the audible alarm. Off by default — click once to enable before the session. |
| **Database →** | Navigates to the database manager. |

### Room Connections panel

A live grid of cards — one per room that has ever sent a heartbeat ping. Cards appear automatically on first ping, no configuration needed.

| Card state | Appearance | Meaning |
|---|---|---|
| **Connected** | Green border, pulsing dot | Pi pinged within the last 3 seconds. |
| **Lost** | Red border (animated pulse), solid red dot | No ping for >3 seconds. |

The `N / N online` badge turns red as soon as any room loses connection. The "last seen" timestamp updates every 500 ms.

**Fail-safe:** if a Pi goes offline, its card turns red within 3 seconds regardless of the server or other rooms.

### Alert cards

Color-coded by type. Each card shows room number, alert type, server-assigned ID, human-readable name, and timestamp.

The **Dismiss** button marks the alert resolved in the database, removes the card on all open tabs, and stops the alarm beep if no other active alerts remain.

---

## Database Manager (`db.html`)

Accessible at `http://localhost:5000/db` or via **Database →**.

### Alerts tab

Full CRUD on the `alerts` table — Add, Edit, Delete.

### Dismiss Log tab

Read and delete operations on the audit log.

---

## Pi Simulator (`pi_simulator.py`)

Impersonates one or more Raspberry Pis for development and testing. Not needed on the actual Pi.

```bash
# Single room (default room 1)
python pi_simulator.py

# Multiple rooms simultaneously
python pi_simulator.py --rooms 1,2,3

# Auto-mode: random alert every 10 s
python pi_simulator.py --rooms 1,2,3 --auto

# Point at a server on another machine
python pi_simulator.py --rooms 1,2 --url http://192.168.1.10:5000
```

### Interactive commands

| Command | Action |
|---|---|
| `1` | Fire alert to default room |
| `1 2` | Fire alert to room 2 |
| `2` | Unauthorized Presence alert |
| `3` | Unauthorized Touch alert |
| `r` | Random alert, random room |
| `r 3` | Random alert to room 3 |
| `q` | Quit |

---

## File Reference

| File | Purpose |
|---|---|
| `app.py` | Flask application. All HTTP routes and SocketIO event handlers. Detects frozen (packaged) mode and adjusts asset paths accordingly. Auto-opens browser when run as exe. |
| `database.py` | All SQLite operations. Stores `ward_watcher.db` next to the source file in development, or next to the exe when packaged. |
| `pi_simulator.py` | Debug simulator. One heartbeat thread per room, interactive alert menu. |
| `pi_client.py` | Runs on the Raspberry Pi. Sends heartbeat pings and alerts to the server. |
| `ward_watcher.spec` | PyInstaller spec — defines what to bundle (templates, static, hidden imports). |
| `build_windows.bat` | One-click build script. Activates venv, installs PyInstaller, runs the spec. |
| `templates/index.html` | Dashboard markup — no inline CSS or JS. |
| `templates/db.html` | Database manager markup — no inline CSS or JS. |
| `static/css/index.css` | Dashboard styles: header, room cards, alert cards, color themes, responsive layout. |
| `static/css/db.css` | Database manager styles: tabs, table, modal, toast, form. |
| `static/js/index.js` | Dashboard logic: per-room heartbeat tracking, room card rendering, Web Audio alarm, SocketIO listeners. |
| `static/js/db.js` | Database manager logic: tab switching, CRUD calls, modal, toast notifications. |
| `static/js/socket.io.min.js` | Socket.IO 4.7.5 client library — bundled locally for fully offline operation. |
| `ward_watcher.db` | SQLite database (auto-created on first run). Contains `alerts` and `dismiss_log` tables. |
