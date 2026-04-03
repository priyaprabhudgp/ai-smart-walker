from picamera2 import Picamera2
from picamera2.outputs import FileOutput
from picamera2.encoders import H264Encoder
import time

picam2 = Picamera2()

# --- 1. PHOTO ---
print("Capturing 12MP Photo...")
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()
picam2.capture_file("test_photo.jpg")
picam2.stop()
print("Photo saved.")
time.sleep(1)

# --- 2. VIDEO ---
print("Reconfiguring for High-Res Video...")

# Key fix: explicitly set the 'encode' key so picamera2 knows which
# stream to feed into the H264 encoder. Without this it can fail silently.
video_config = picam2.create_video_configuration(
    main={"size": (2304, 1296), "format": "RGB888"},
    encode="main"   # <-- tells the encoder which stream to use
)
picam2.configure(video_config)

encoder = H264Encoder(bitrate=10_000_000)
output = FileOutput("test_video.h264")

try:
    print("Starting Recording...")
    # Do NOT call picam2.start() before start_recording().
    # start_recording() starts the camera internally.
    picam2.start_recording(encoder, output)
    print("Recording... (5 seconds)")
    time.sleep(5)
    picam2.stop_recording()
    print("Video Captured!")
except Exception as e:
    print(f"Recording failed: {e}")
finally:
    picam2.stop()
