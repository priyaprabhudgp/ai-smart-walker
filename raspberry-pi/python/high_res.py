from picamera2 import Picamera2

picam2 = Picamera2()

# 1. Configure for a High-Res Still (12MP)
# On Bookworm, it often defaults to the maximum sensor resolution
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()

# Capture the 12MP Photo
picam2.capture_file("pro_photo.jpg")
print("Photo captured!")

# 2. Switch to Video
# To change resolution for video, we update the configuration
video_config = picam2.create_video_configuration()
# Manually setting a high-res video window (e.g., 2304x1296)
video_config.main.size = (2304, 1296)
picam2.configure(video_config)

picam2.start_recording("pro_video.h264")
print("Recording video...")
picam2.wait(5) 
picam2.stop_recording()

picam2.stop()
