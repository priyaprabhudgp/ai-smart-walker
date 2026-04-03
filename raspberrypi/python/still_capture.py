from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start()
picam2.capture_file("image_12mp.jpg")
picam2.stop()
