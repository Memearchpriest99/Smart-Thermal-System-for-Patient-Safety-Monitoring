  import requests, threading, time
  from datetime import datetime, timezone

  SERVER  = "http://10.243.12.176:5000"
  ROOM_ID = 1

  ALERT_TYPES = {
      "1": "fire",
      "2": "unauthorized_presence",
      "3": "unauthorized_touch",
  }

  def heartbeat_loop():
      while True:
          try:
              requests.post(f"{SERVER}/api/ping",
                            json={"room_id": ROOM_ID}, timeout=2)
          except Exception:
              pass
          time.sleep(2)

  def send_alert(alert_type):
      try:
          r = requests.post(f"{SERVER}/alert", json={
              "alert_type": alert_type,
              "room_id":    ROOM_ID,
              "timestamp":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
          }, timeout=5)
          print(f"  Sent: {alert_type} -> alert_id {r.json().get('alert_id')}")
      except Exception as e:
          print(f"  Failed: {e}")

  threading.Thread(target=heartbeat_loop, daemon=True).start()
  print(f"Heartbeat running -> {SERVER}  (Room {ROOM_ID})")
  print("1=Fire  2=Unauthorized Presence  3=Unauthorized Touch  q=Quit")

  while True:
      cmd = input("Alert > ").strip().lower()
      if cmd in ALERT_TYPES:
          send_alert(ALERT_TYPES[cmd])
      elif cmd == "q":
          break
