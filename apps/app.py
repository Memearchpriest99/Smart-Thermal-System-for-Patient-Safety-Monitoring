"""
Ward Watcher — Flask-SocketIO surveillance server.

Endpoints
---------
POST /alert              Accept alert from sensor, store in DB, broadcast to clients.
POST /api/ping           Receive a heartbeat from the Pi; relay to all dashboard clients.
GET  /db                 Serve the database management interface.
GET  /api/alerts         List all alerts (JSON).
POST /api/alerts         Create an alert via the DB UI.
PUT  /api/alerts/<id>    Update an alert.
DELETE /api/alerts/<id>  Delete an alert.
GET  /api/dismiss-log    List all dismiss log entries (JSON).
DELETE /api/dismiss-log/<id>  Delete a dismiss log entry.

SocketIO events
---------------
  server → client   room_heartbeat   {"room_id": int, "server_time": float}
  server → client   new_alert        {alert_id, alert_type, room_id, timestamp}
  server → client   alert_dismissed  {"alert_id": int}
  client → server   dismiss_alert    {"alert_id": int}

Heartbeat design
----------------
Each room's Pi calls POST /api/ping (with its room_id) every ~2 seconds.
The server relays a 'room_heartbeat' event to all dashboard clients.
The dashboard tracks each room independently: if a room's pings stop for
more than 3 seconds its card turns red ('Lost'). Multiple rooms are fully
supported — each appears as its own card in the Room Connections panel.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

import database as db

# When frozen by PyInstaller, Flask needs explicit paths to bundled assets.
if getattr(sys, "frozen", False):
    _base = sys._MEIPASS
    app = Flask(
        __name__,
        template_folder=os.path.join(_base, "templates"),
        static_folder=os.path.join(_base, "static"),
    )
else:
    app = Flask(__name__)

app.config["SECRET_KEY"] = "ward-watcher-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ------------------------------------------------------------------ #
# Pi heartbeat relay                                                   #
# ------------------------------------------------------------------ #

@app.route("/api/ping", methods=["POST"])
def pi_ping():
    """
    Receive a per-room liveness ping from a Raspberry Pi and relay it
    to all connected dashboard clients as a 'room_heartbeat' event.

    Expected JSON body: {"room_id": int}

    Each room's Pi calls this every ~2 seconds. The dashboard tracks
    each room independently and marks a room 'Lost' when its pings stop.
    """
    payload: dict = request.get_json(force=True, silent=True) or {}
    try:
        room_id = int(payload["room_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "room_id must be an integer"}), 400

    socketio.emit("room_heartbeat", {
        "room_id":     room_id,
        "server_time": datetime.now(timezone.utc).timestamp(),
    })
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------------------ #
# Sensor endpoint                                                      #
# ------------------------------------------------------------------ #

@app.route("/alert", methods=["POST"])
def receive_alert():
    """
    Accept a raw alert from a sensor device.

    Expected JSON body:
        {
            "alert_type": "fire" | "unauthorized_presence" | "unauthorized_touch",
            "room_id":    int,
            "timestamp":  str   (ISO 8601; defaults to server time if omitted)
        }

    Returns the generated alert packet including server-assigned alert_id.
    """
    payload: dict = request.get_json(force=True, silent=True) or {}

    alert_type = payload.get("alert_type", "")
    if alert_type not in db.VALID_ALERT_TYPES:
        return jsonify({
            "error": f"alert_type must be one of: {', '.join(sorted(db.VALID_ALERT_TYPES))}"
        }), 400

    try:
        room_id = int(payload["room_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "room_id must be an integer"}), 400

    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    alert_id = db.insert_alert(alert_type, room_id, timestamp)

    packet = {
        "alert_id":   alert_id,
        "alert_type": alert_type,
        "room_id":    room_id,
        "timestamp":  timestamp,
    }
    socketio.emit("new_alert", packet)
    print(
        f"[ALERT #{alert_id}] Room {room_id} | {alert_type.upper()} | {timestamp}"
    )
    return jsonify({"status": "ok", "alert_id": alert_id}), 201


# ------------------------------------------------------------------ #
# Pages                                                                #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    """Serve the live surveillance dashboard."""
    return render_template("index.html")


@app.route("/db")
def db_view():
    """Serve the database management interface."""
    return render_template("db.html")


# ------------------------------------------------------------------ #
# Alerts REST API (for DB interface)                                   #
# ------------------------------------------------------------------ #

@app.route("/api/alerts", methods=["GET"])
def api_list_alerts():
    return jsonify(db.get_all_alerts())


@app.route("/api/alerts", methods=["POST"])
def api_create_alert():
    """Create an alert through the DB UI (bypasses SocketIO broadcast)."""
    payload: dict = request.get_json(force=True, silent=True) or {}
    alert_type = payload.get("alert_type", "")
    if alert_type not in db.VALID_ALERT_TYPES:
        return jsonify({"error": "Invalid alert_type"}), 400
    try:
        room_id = int(payload["room_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "room_id must be an integer"}), 400
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alert_id = db.insert_alert(alert_type, room_id, timestamp)
    return jsonify({"status": "ok", "alert_id": alert_id}), 201


@app.route("/api/alerts/<int:alert_id>", methods=["PUT"])
def api_update_alert(alert_id: int):
    payload: dict = request.get_json(force=True, silent=True) or {}
    if not db.update_alert(alert_id, payload):
        return jsonify({"error": "No valid fields or alert not found"}), 400
    return jsonify({"status": "ok"})


@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
def api_delete_alert(alert_id: int):
    db.delete_alert(alert_id)
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------ #
# Dismiss log REST API                                                 #
# ------------------------------------------------------------------ #

@app.route("/api/dismiss-log", methods=["GET"])
def api_list_dismiss_log():
    return jsonify(db.get_all_dismiss_logs())


@app.route("/api/dismiss-log/<int:log_id>", methods=["DELETE"])
def api_delete_dismiss_log(log_id: int):
    db.delete_dismiss_log(log_id)
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------ #
# SocketIO events                                                      #
# ------------------------------------------------------------------ #

@socketio.on("connect")
def on_connect():
    print(f"[CONNECT]    {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    print(f"[DISCONNECT] {request.sid}")


@socketio.on("dismiss_alert")
def on_dismiss(data: dict):
    """
    Client requests dismissal of a specific alert.

    Expected: {"alert_id": int}
    Broadcasts "alert_dismissed" to all clients on success.
    """
    alert_id = data.get("alert_id")
    if not isinstance(alert_id, int):
        return

    dismissed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    success = db.dismiss_alert(alert_id, dismissed_at)

    if success:
        alert = db.get_alert(alert_id)
        print(
            f"[DISMISS #{alert_id}] Room {alert['room_id']} "
            f"| {alert['alert_type']} | dismissed at {dismissed_at}"
        )
        socketio.emit("alert_dismissed", {"alert_id": alert_id})


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def _open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open_new_tab("http://localhost:5000")


if __name__ == "__main__":
    _frozen = getattr(sys, "frozen", False)
    db.init_db()
    print("[SERVER] Ward Watcher starting → http://localhost:5000")
    if _frozen:
        threading.Thread(target=_open_browser, daemon=True).start()
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    else:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True)
