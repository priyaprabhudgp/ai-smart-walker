"""
ai/ultrasonic.py

Reads distance from up to 4 HC-SR04 ultrasonic sensors.
Returns distances in meters, keyed by position: "left", "right", "front", "down"

Wiring (BCM pin numbering):
    Left  sensor: TRIG=23, ECHO=24
    Right sensor: TRIG=17, ECHO=27
    Front sensor: TRIG=5,  ECHO=6
    Down  sensor: TRIG=13, ECHO=19

Usage:
    from ultrasonic import UltrasonicArray
    sensors = UltrasonicArray()
    distances = sensors.read_all()
    # returns e.g. {"left": 0.42, "right": 1.1, "front": None, "down": None}
    sensors.cleanup()
"""

import time
from typing import Optional

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# ----- SENSOR CONFIG -----

SENSORS: dict[str, dict] = {
    "left":  {"trig": 23, "echo": 24},
    "right": {"trig": 17, "echo": 27},
    "front": {"trig": 5,  "echo": 6},
    "down":  {"trig": 13, "echo": 19},
}

TIMEOUT = 0.04   # max seconds to wait for echo (covers ~6m range)
SPEED_OF_SOUND = 17150  # cm/s at ~20°C


# ----- SENSOR CLASS -----

class UltrasonicArray:
    """
    Manages multiple HC-SR04 sensors.
    Only initializes sensors that have both TRIG and ECHO pins set.
    """

    def __init__(self):
        self._active = {}  # position -> pin config

        if not GPIO_AVAILABLE:
            print("[UltrasonicArray] RPi.GPIO not available -- returning None for all distances.")
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for position, pins in SENSORS.items():
            if pins["trig"] is not None and pins["echo"] is not None:
                GPIO.setup(pins["trig"], GPIO.OUT)
                GPIO.setup(pins["echo"], GPIO.IN)
                GPIO.output(pins["trig"], False)
                self._active[position] = pins
                print(f"[UltrasonicArray] {position} sensor ready (TRIG={pins['trig']} ECHO={pins['echo']})")

        time.sleep(0.5)  # let sensors settle

    def read(self, position: str) -> Optional[float]:
        """
        Read distance from a single sensor.
        Returns distance in meters, or None if sensor not wired or read failed.
        """
        if position not in self._active:
            return None

        pins = self._active[position]
        trig = pins["trig"]
        echo = pins["echo"]

        # Send 10us pulse
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        # Wait for echo to go high (sound leaves sensor)
        start = time.time()
        while GPIO.input(echo) == 0:
            if time.time() - start > TIMEOUT:
                return None
            pulse_start = time.time()

        # Wait for echo to go low (sound comes back)
        start = time.time()
        while GPIO.input(echo) == 1:
            if time.time() - start > TIMEOUT:
                return None
            pulse_end = time.time()

        distance_cm = (pulse_end - pulse_start) * SPEED_OF_SOUND #17150
        distance_m  = round(distance_cm / 100, 2)

        # Sanity check:  HC-SR04 range is 2cm to 400cm
        if distance_m < 0.02 or distance_m > 4.0:
            return None

        return distance_m

    def read_all(self) -> dict[str, Optional[float]]:
        """
        Read all sensors. Returns dict of position -> distance in meters (or None).
        Example: {"left": 0.42, "right": 1.1, "front": None, "down": None}
        """
        return {position: self.read(position) for position in SENSORS}

    def cleanup(self):
        """Release GPIO pins. Call when shutting down."""
        if GPIO_AVAILABLE:
            GPIO.cleanup()





