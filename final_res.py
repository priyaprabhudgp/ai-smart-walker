from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import subprocess
import time

picam2 = Picamera2()

# --- PHOTO at full 12MP ---
print("Capturing 12MP Photo...")
picam2.configure(picam2.create_still_configuration())
picam2.start()
picam2.capture_file("photo.jpg")
picam2.stop()
print("Photo saved: photo.jpg")
time.sleep(1)

# --- VIDEO at 1080p (hardware H264 ceiling on Pi 4 / IMX708) ---
print("Configuring 1080p Video...")
video_config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "YUV420"},
    encode="main"
)
picam2.configure(video_config)

try:
    print("Recording 5 seconds...")
    picam2.start_recording(
        H264Encoder(bitrate=15_000_000),
        FileOutput("video.h264")
    )
    time.sleep(5)
    picam2.stop_recording()
    print("Raw H264 saved: video.h264")

    # Wrap in MP4 container (no re-encode, instant)
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", "video.h264",
        "-c", "copy",
        "video.mp4"
    ], check=True)
    print("Done! video.mp4 ready.")

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    picam2.stop()
