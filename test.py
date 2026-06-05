from pythonosc.udp_client import SimpleUDPClient
import time
import random

# ============================================
# CONFIG
# ============================================

PD_IP = "127.0.0.1"
PD_PORT = 9000

client = SimpleUDPClient(PD_IP, PD_PORT)

# ============================================
# TEST LOOP
# ============================================

gestures = [
    "fist",
    "peace",
    "open_hand",
    "thumbs_up"
]

print("Sending OSC messages to Pure Data...\n")

while True:

    # random gesture
    gesture = random.choice(gestures)

    # fake sensitivity/confidence
    sensitivity = round(random.uniform(0.5, 1.0), 2)

    # send OSC
    client.send_message(
        "/gesture",
        [gesture, sensitivity]
    )

    print(
        f"Sent -> gesture: {gesture} | sensitivity: {sensitivity}"
    )

    time.sleep(1)