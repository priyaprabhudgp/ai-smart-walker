











































pipriya@rasperrypi:~ $ 
pipriya@rasperrypi:~ $ sudo nano /boot/firmware/config.txt
pipriya@rasperrypi:~ $ sudo reboot

Broadcast message from root@rasperrypi on pts/2 (Tue 2026-03-31 18:44:06 PDT):

The system will reboot now!

pipriya@rasperrypi:~ $ Connection to 192.168.1.117 closed by remote host.
Connection to 192.168.1.117 closed.
prabhuvenkatesan@Prabhus-MacBook-Air ~ % ssh pipriya@192.168.1.117
pipriya@192.168.1.117's password: 
Linux rasperrypi 6.12.75+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.12.75-1+rpt1~bookworm (2026-03-11) aarch64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
Last login: Tue Mar 31 18:44:17 2026
pipriya@rasperrypi:~ $ python3 -c "import numpy; print('numpy:', numpy.__version__)"
# Must show 1.26.x

python3 -c "import onnxruntime; print('onnxruntime:', onnxruntime.__version__)"
# Must show 1.x.x

python3 -c "import cv2; print('opencv:', cv2.__version__)"
# Must show 4.6.x

python3 -c "from picamera2 import Picamera2; print('picamera2 OK')"
# Must show picamera2 OK
numpy: 1.26.4
2026-03-31 18:50:19.488151356 [W:onnxruntime:Default, device_discovery.cc:325 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:92 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
onnxruntime: 1.24.4
opencv: 4.6.0
picamera2 OK
pipriya@rasperrypi:~ $ # Install numpy 1.x (2.x breaks opencv and onnxruntime)
pip3 install "numpy==1.26.4" --break-system-packages

# Install onnxruntime (replaces torch - Pi native, no ARM issues)
pip3 install onnxruntime --break-system-packages

# Fix PATH so local scripts are accessible
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Requirement already satisfied: numpy==1.26.4 in ./.local/lib/python3.11/site-packages (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Requirement already satisfied: onnxruntime in ./.local/lib/python3.11/site-packages (1.24.4)
Requirement already satisfied: flatbuffers in ./.local/lib/python3.11/site-packages (from onnxruntime) (20181003210633)
Requirement already satisfied: numpy>=1.21.6 in ./.local/lib/python3.11/site-packages (from onnxruntime) (1.26.4)
Requirement already satisfied: packaging in /usr/lib/python3/dist-packages (from onnxruntime) (23.0)
Requirement already satisfied: protobuf in ./.local/lib/python3.11/site-packages (from onnxruntime) (7.34.1)
Requirement already satisfied: sympy in /usr/lib/python3/dist-packages (from onnxruntime) (1.11.1)
pipriya@rasperrypi:~ $ # Find working download URL
rm -f yolov8n.onnx
wget -O yolov8n.onnx "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx" || \
wget -O yolov8n.onnx "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.onnx" || \
wget -O yolov8n.onnx "https://storage.googleapis.com/ailia-models/yolov8/yolov8n.onnx"

ls -lh yolov8n.onnx
# Must be ~6MB, not 0 bytes




cat > ~/yolo_detect.py << 'EOF'
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')  # system opencv

import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

# COCO class names
CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

def preprocess(frame, input_size=640):
    """Resize and normalize frame for YOLOv8"""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    # Pad to square
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
python3 ~/yolo_detect.pyrames: {count} | Avg FPS: {count/elapsed:.1f}")"cls{cls_id}"
--2026-03-31 18:51:30--  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
Resolving github.com (github.com)... 140.82.116.3
Connecting to github.com (github.com)|140.82.116.3|:443... connected.
HTTP request sent, awaiting response... 404 Not Found
2026-03-31 18:51:30 ERROR 404: Not Found.

--2026-03-31 18:51:30--  https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.onnx
Resolving github.com (github.com)... 140.82.116.3
Connecting to github.com (github.com)|140.82.116.3|:443... connected.
HTTP request sent, awaiting response... 404 Not Found
2026-03-31 18:51:31 ERROR 404: Not Found.

--2026-03-31 18:51:31--  https://storage.googleapis.com/ailia-models/yolov8/yolov8n.onnx
Resolving storage.googleapis.com (storage.googleapis.com)... 2607:f8b0:4002:c02::cf, 2607:f8b0:4002:c0f::cf, 2607:f8b0:4002:c11::cf, ...
Connecting to storage.googleapis.com (storage.googleapis.com)|2607:f8b0:4002:c02::cf|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 12673843 (12M) [application/octet-stream]
Saving to: 'yolov8n.onnx'

yolov8n.onnx                    100%[=======================================================>]  12.09M  2.47MB/s    in 5.1s    

2026-03-31 18:51:37 (2.38 MB/s) - 'yolov8n.onnx' saved [12673843/12673843]

-rw-r--r-- 1 pipriya pipriya 13M Mar 14  2023 yolov8n.onnx
2026-03-31 18:51:40.453433679 [W:onnxruntime:Default, device_discovery.cc:325 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:92 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
Loading YOLOv8n ONNX model...
Model loaded. Input: images
Starting camera...
[0:07:17.431450770] [1931]  INFO Camera camera_manager.cpp:330 libcamera v0.5.2+99-bfd68f78
[0:07:17.455675635] [1940]  INFO IPAProxy ipa_proxy.cpp:180 Using tuning file /usr/share/libcamera/ipa/rpi/vc4/ov5647.json
[0:07:17.461905134] [1940]  INFO Camera camera_manager.cpp:220 Adding camera '/base/soc/i2c0mux/i2c@1/ov5647@36' for pipeline handler rpi/vc4
[0:07:17.462019375] [1940]  INFO RPI vc4.cpp:440 Registered camera /base/soc/i2c0mux/i2c@1/ov5647@36 to Unicam device /dev/media2 and ISP device /dev/media1
[0:07:17.462078653] [1940]  INFO RPI pipeline_base.cpp:1107 Using configuration file '/usr/share/libcamera/pipeline/rpi/vc4/rpi_apps.yaml'
[0:07:17.469240281] [1931]  INFO Camera camera.cpp:1215 configuring streams: (0) 640x480-RGB888/SMPTE170M/Rec709/None/Full (1) 640x480-SGBRG10_CSI2P/RAW
[0:07:17.469775077] [1940]  INFO RPI vc4.cpp:615 Sensor: /base/soc/i2c0mux/i2c@1/ov5647@36 - Selected sensor format: 640x480-SGBRG10_1X10/RAW - Selected unicam format: 640x480-pGAA/RAW
Camera ready - detecting... Ctrl+C to stop
  person: 65% [201,17,640,467]
  person: 78% [197,17,639,468]
  person: 79% [203,16,640,468]
  person: 54% [196,15,640,469]
  person: 75% [196,17,639,468]
  person: 77% [202,17,640,467]
  person: 58% [193,17,640,467]
  person: 74% [194,15,639,468]
  person: 79% [201,15,640,466]
  person: 77% [183,10,640,470]
  person: 72% [183,10,639,469]
  person: 85% [187,9,638,470]
  person: 85% [186,9,639,469]
  person: 83% [183,9,640,469]
  person: 82% [186,11,638,469]
  person: 82% [185,11,639,468]
  person: 79% [184,11,640,468]
  person: 82% [181,8,640,467]
  person: 61% [187,6,640,466]
  person: 77% [184,9,640,470]
  person: 71% [184,9,639,470]
  person: 83% [187,6,638,470]
  person: 87% [188,6,639,470]
  person: 85% [185,6,640,470]
  person: 74% [181,7,638,469]
  person: 81% [184,7,639,468]
  person: 79% [185,7,640,469]
  person: 80% [178,7,640,467]
  person: 72% [187,6,640,468]
  person: 75% [183,5,638,467]
  person: 77% [179,5,640,467]
  person: 72% [185,5,639,467]
  person: 87% [182,3,638,468]
  person: 86% [181,3,639,467]
  person: 86% [185,4,640,468]
  person: 84% [182,3,639,469]
  person: 84% [179,3,639,468]
  person: 83% [184,4,640,468]
  person: 84% [177,3,640,469]
  person: 65% [150,10,630,461]
  person: 56% [155,11,636,462]
  person: 74% [153,11,635,462]
  person: 71% [154,10,638,462]
  person: 70% [142,9,639,462]
  person: 74% [151,12,638,466]
  person: 78% [154,12,638,464]
  person: 76% [142,11,640,464]
  person: 72% [145,8,637,468]
  person: 79% [155,7,638,466]
  person: 80% [145,5,639,457]
  person: 75% [137,6,640,462]
  person: 85% [146,8,638,462]
  person: 88% [148,6,639,462]
  person: 85% [142,5,639,465]
  person: 83% [146,9,638,466]
  person: 85% [145,8,639,467]
  person: 82% [137,9,640,468]
  person: 83% [141,6,639,469]
  person: 79% [136,6,640,468]
  person: 83% [134,4,639,457]
  person: 81% [136,5,639,461]
  person: 68% [140,4,636,452]
  person: 84% [132,1,638,457]
  person: 85% [132,1,639,460]
  person: 71% [121,2,639,466]
  person: 81% [126,3,638,464]
  person: 82% [128,3,638,466]
  person: 74% [124,5,638,466]
  person: 73% [105,8,640,469]
  person: 69% [90,8,640,473]
  person: 79% [76,6,639,467]
  person: 81% [105,6,639,471]
  person: 82% [85,6,639,473]
  person: 76% [71,8,639,469]
  person: 78% [97,8,639,472]
  person: 83% [80,7,639,471]
  person: 74% [72,6,639,473]
  person: 79% [68,5,639,470]
  person: 70% [50,14,639,457]
  person: 71% [47,13,639,459]
  person: 81% [36,15,637,464]
  person: 80% [41,14,639,461]
  person: 77% [29,13,638,463]
  person: 80% [32,16,637,464]
  person: 79% [47,15,639,466]
  person: 79% [38,14,639,467]
  person: 73% [62,9,638,466]
  person: 65% [77,26,624,458]
  person: 65% [103,26,637,464]
  person: 61% [11,22,632,462]
  person: 65% [54,22,630,457]
  person: 70% [89,24,636,462]
  person: 72% [6,19,635,464]
  person: 73% [33,19,637,463]
  person: 76% [70,19,638,468]
  person: 73% [4,37,637,469]
  person: 72% [9,37,639,469]
  person: 78% [5,31,633,467]
  person: 78% [3,30,637,472]
  person: 81% [6,32,637,473]
  person: 81% [4,33,636,470]
  person: 81% [4,33,638,472]
  person: 79% [4,32,638,470]
  person: 73% [4,31,640,471]
  person: 73% [3,35,638,472]
  person: 74% [10,35,639,472]
  person: 70% [2,29,636,468]
  person: 78% [1,29,638,472]
  person: 82% [7,30,637,472]
  person: 73% [1,29,637,471]
  person: 80% [2,28,638,472]
  person: 80% [4,27,638,471]
  person: 80% [4,25,640,471]
  person: 52% [1,0,639,468]
  person: 55% [3,0,638,468]
  person: 58% [1,0,640,467]
  person: 63% [0,6,639,466]
  person: 63% [2,7,638,469]
  person: 62% [0,9,639,468]
  person: 69% [2,17,638,472]
  person: 65% [0,18,639,471]
  person: 53% [4,21,637,471]
  person: 56% [3,2,638,467]
  person: 57% [6,3,640,468]
--- FPS: 1.6 | Frame: 20 ---
  keyboard: 90% [0,2,623,471]
  keyboard: 92% [0,1,622,470]
  keyboard: 91% [1,1,610,470]
  keyboard: 68% [1,0,628,473]
  keyboard: 92% [1,0,625,473]
  keyboard: 91% [0,0,621,474]
  keyboard: 85% [0,1,607,476]
  keyboard: 55% [0,1,624,477]
  keyboard: 62% [1,1,616,476]
  mouse: 76% [521,166,639,354]
  mouse: 68% [521,167,639,353]
  mouse: 69% [520,168,639,356]
  mouse: 84% [520,168,639,357]
  mouse: 72% [520,168,639,355]
  mouse: 68% [520,167,639,359]
  mouse: 52% [520,167,639,358]
  keyboard: 66% [0,0,633,471]
  keyboard: 51% [0,0,638,471]
  keyboard: 88% [0,0,634,471]
  keyboard: 90% [0,0,630,472]
  keyboard: 73% [1,0,629,472]
  keyboard: 93% [1,0,631,473]
  keyboard: 92% [0,0,624,473]
  keyboard: 78% [1,0,613,475]
  keyboard: 86% [0,0,610,473]
  keyboard: 62% [2,0,598,475]
  mouse: 53% [536,165,639,376]
  keyboard: 71% [0,0,638,469]
  keyboard: 50% [0,0,638,468]
  keyboard: 91% [1,1,639,467]
  keyboard: 94% [0,1,638,466]
  keyboard: 85% [1,1,638,465]
  keyboard: 94% [0,1,638,471]
  keyboard: 96% [0,1,638,470]
  keyboard: 91% [1,0,637,471]
  keyboard: 80% [0,0,635,473]
  keyboard: 91% [0,0,635,472]
  keyboard: 71% [1,0,635,474]
  keyboard: 73% [1,0,637,470]
  keyboard: 89% [2,1,636,470]
  keyboard: 93% [2,1,637,469]
  keyboard: 88% [3,1,638,468]
  keyboard: 93% [1,0,636,471]
  keyboard: 95% [1,0,637,470]
  keyboard: 92% [3,0,638,471]
  keyboard: 84% [0,0,637,475]
  keyboard: 92% [0,0,637,475]
  keyboard: 68% [3,0,638,475]
  keyboard: 78% [0,0,640,473]
  keyboard: 67% [0,0,638,473]
  keyboard: 86% [0,0,636,473]
  keyboard: 92% [0,1,639,472]
  keyboard: 89% [1,1,637,471]
  keyboard: 89% [0,0,637,473]
  keyboard: 94% [0,0,640,473]
  keyboard: 93% [2,0,637,473]
  keyboard: 89% [0,0,639,475]
  keyboard: 83% [1,0,638,475]
  keyboard: 73% [0,0,639,471]
  keyboard: 85% [0,0,636,473]
  keyboard: 91% [0,0,638,472]
  keyboard: 92% [1,0,638,472]
  keyboard: 91% [0,0,635,471]
  keyboard: 94% [0,0,639,472]
  keyboard: 94% [0,0,637,472]
  keyboard: 72% [0,1,637,477]
  keyboard: 90% [0,0,638,476]
  keyboard: 85% [0,0,637,477]
  person: 50% [238,288,319,412]
  tv: 51% [317,66,639,379]
  tv: 66% [319,65,639,381]
  tv: 57% [303,68,639,416]
  tv: 59% [303,68,639,413]
  tv: 66% [302,68,639,410]
  tv: 53% [302,58,639,416]
  bird: 58% [224,284,305,410]
  bird: 58% [224,284,306,410]
  bird: 55% [224,284,306,410]
  bird: 61% [224,285,305,410]
  bird: 51% [224,285,305,410]
  bird: 55% [223,285,305,410]
  laptop: 56% [284,69,639,426]
  laptop: 71% [257,65,639,432]
  laptop: 65% [294,128,638,461]
  laptop: 69% [294,128,639,459]
  laptop: 63% [293,128,639,461]
  laptop: 80% [295,128,638,463]
  laptop: 76% [295,128,639,459]
  laptop: 72% [294,128,639,458]
  laptop: 52% [295,116,639,455]
  bird: 54% [209,317,291,445]
  bird: 56% [210,317,292,446]
  bird: 58% [209,317,292,446]
  bird: 64% [209,317,292,446]
  bird: 61% [209,317,292,446]
  bird: 53% [209,317,292,446]
  bird: 56% [209,318,292,446]
  tv: 50% [290,103,639,440]
  cat: 53% [265,276,491,475]
--- FPS: 1.6 | Frame: 40 ---
  person: 56% [555,346,639,478]
  person: 54% [557,346,639,477]
  person: 68% [555,347,639,478]
  person: 64% [556,347,639,478]
  person: 55% [556,346,639,478]
  person: 59% [555,346,639,478]
  person: 65% [556,347,639,478]
  person: 53% [556,346,639,478]
  person: 62% [554,346,639,476]
  bird: 55% [159,367,289,478]
  bird: 65% [159,367,288,478]
  bird: 51% [158,367,289,478]
  bird: 63% [159,367,288,478]
  bird: 51% [159,367,288,477]
  person: 59% [562,389,639,478]
  person: 53% [563,389,639,478]
  person: 57% [562,389,639,478]
  person: 56% [563,389,639,478]
  person: 55% [563,389,639,478]
  person: 51% [563,389,639,479]
  toothbrush: 54% [171,17,359,463]
  toothbrush: 61% [249,206,318,336]
  toothbrush: 57% [249,205,317,338]
  toothbrush: 52% [249,206,317,339]
--- FPS: 1.6 | Frame: 60 ---
  toothbrush: 51% [267,195,373,473]
  toothbrush: 52% [261,196,374,474]
--- FPS: 1.6 | Frame: 80 ---
  laptop: 60% [263,323,640,473]
  laptop: 75% [265,323,639,474]
  laptop: 73% [264,323,639,473]
  laptop: 72% [262,323,638,473]
  laptop: 74% [265,324,638,473]
  laptop: 76% [265,324,639,473]
  laptop: 73% [263,324,638,473]
  laptop: 69% [265,323,638,473]
  laptop: 73% [265,323,639,472]
  laptop: 68% [263,324,638,472]
  laptop: 75% [252,297,640,473]
  laptop: 70% [252,297,640,473]
  laptop: 57% [251,297,638,474]
  laptop: 77% [252,297,639,474]
  laptop: 69% [253,297,639,474]
  laptop: 61% [251,297,639,474]
  laptop: 74% [252,297,639,473]
  laptop: 62% [254,298,639,473]
--- FPS: 1.6 | Frame: 100 ---
  laptop: 73% [264,284,639,474]
  laptop: 78% [263,284,639,473]
  laptop: 72% [262,284,638,473]
  laptop: 77% [265,284,639,473]
  laptop: 79% [263,285,638,473]
  laptop: 75% [261,285,638,473]
  laptop: 76% [266,284,638,473]
  laptop: 76% [264,285,638,473]
  laptop: 70% [263,285,638,473]
  laptop: 83% [164,264,638,475]
  laptop: 85% [163,264,638,474]
  laptop: 84% [163,264,638,474]
  laptop: 75% [162,264,640,474]
  laptop: 88% [164,264,638,473]
  laptop: 88% [164,265,638,473]
  laptop: 73% [162,266,640,473]
  laptop: 87% [164,265,638,473]
  laptop: 83% [165,265,637,473]
  laptop: 52% [165,266,639,473]
  laptop: 70% [96,255,573,477]
  laptop: 67% [93,255,574,475]
  laptop: 68% [98,256,574,475]
  laptop: 58% [94,255,573,474]
  laptop: 63% [98,255,573,475]
  laptop: 58% [70,249,553,474]
  laptop: 57% [69,249,556,474]
  laptop: 69% [69,249,553,473]
  laptop: 63% [68,249,554,473]
  laptop: 71% [69,249,553,472]
  laptop: 72% [68,249,554,472]
  laptop: 65% [70,250,552,472]
  laptop: 60% [70,250,552,472]
  laptop: 88% [75,235,572,474]
  laptop: 86% [71,234,572,474]
  laptop: 84% [75,234,575,473]
  laptop: 89% [72,235,571,473]
  laptop: 89% [68,235,572,474]
  laptop: 67% [28,235,569,474]
  laptop: 88% [74,235,571,472]
  laptop: 91% [71,235,571,472]
  tv: 63% [121,110,639,477]
  tv: 75% [120,111,639,477]
  tv: 73% [128,111,639,477]
  tv: 54% [123,110,638,477]
  tv: 70% [122,111,638,477]
  tv: 71% [124,110,638,477]
  tv: 59% [121,109,639,477]
  tv: 56% [121,109,638,477]
  laptop: 51% [127,130,638,471]
  laptop: 63% [123,127,638,473]
  laptop: 52% [82,97,640,475]
  laptop: 55% [91,98,639,472]
  laptop: 55% [79,96,629,474]
  laptop: 55% [77,102,625,475]
  laptop: 61% [100,88,635,468]
  laptop: 65% [111,89,637,468]
  laptop: 57% [75,86,636,471]
  laptop: 59% [104,87,638,471]
  toothbrush: 67% [0,0,373,312]
  toothbrush: 58% [0,0,375,314]
--- FPS: 1.6 | Frame: 120 ---
  scissors: 59% [1,2,636,471]
  scissors: 69% [1,8,638,471]
  scissors: 67% [0,5,638,471]
  scissors: 69% [0,5,638,474]
  person: 53% [2,14,514,474]
  person: 55% [5,12,500,473]
  person: 63% [1,11,531,472]
  person: 66% [2,13,509,472]
  person: 54% [0,14,491,473]
  person: 62% [238,92,557,470]
  person: 66% [238,92,560,470]
  person: 65% [230,92,557,471]
  person: 62% [228,92,562,471]
  person: 60% [237,92,560,470]
  person: 66% [227,90,556,473]
  person: 63% [229,90,561,473]
  person: 54% [236,91,561,473]
  person: 58% [230,90,555,473]
  person: 54% [236,91,559,474]
  person: 58% [255,76,585,472]
  person: 72% [248,75,582,471]
  person: 77% [248,75,591,471]
  person: 69% [251,75,590,470]
  person: 76% [234,75,584,473]
  person: 74% [234,76,590,473]
  person: 59% [242,76,589,472]
  person: 74% [238,74,588,474]
  person: 69% [239,74,590,474]
  person: 51% [0,0,545,473]
  person: 57% [0,2,540,475]
  chair: 59% [529,242,639,475]
  chair: 64% [528,242,639,474]
  chair: 67% [528,242,639,475]
  chair: 62% [530,243,639,474]
  chair: 55% [529,242,639,475]
  chair: 54% [528,242,639,475]
  person: 61% [4,0,566,470]
  person: 50% [2,3,589,469]
  person: 62% [5,2,555,471]
--- FPS: 1.6 | Frame: 140 ---
  person: 54% [5,114,465,475]
  person: 59% [2,113,479,474]
  person: 64% [6,111,484,473]
  person: 62% [2,112,455,474]
  person: 58% [0,112,453,474]
  person: 64% [8,111,463,474]
  person: 60% [42,74,372,472]
  person: 70% [37,76,413,475]
  person: 73% [49,76,416,475]
  person: 64% [38,78,419,475]
  person: 75% [42,75,415,475]
  person: 76% [50,75,416,475]
  person: 67% [45,74,418,476]
  person: 65% [60,97,384,471]
  person: 72% [59,96,385,469]
  person: 55% [57,96,390,473]
  person: 58% [53,95,389,472]
  person: 53% [44,97,399,474]
  person: 57% [56,96,410,474]
  person: 69% [48,95,408,474]
  person: 56% [56,93,407,474]
  person: 66% [45,92,404,474]
  person: 57% [59,92,406,475]
^C
Stopped. Frames: 143 | Avg FPS: 1.6
pipriya@rasperrypi:~ $ cat > ~/yolo_detect.py << 'EOF'
pipriya@rasperrypi:~ $ python3 ~/yolo_detect.py
pipriya@rasperrypi:~ $ sudo nano ~/yolo_detect.py












































  GNU nano 7.2                                      /home/pipriya/yolo_detect.py                                                
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')  # system opencv

import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

# COCO class names
CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

def preprocess(frame, input_size=640):
    """Resize and normalize frame for YOLOv8"""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    # Pad to square
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    # Normalize and transpose to NCHW
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, nh, nw

def postprocess(outputs, scale, orig_h, orig_w, conf_thresh=0.5):
    """Parse YOLOv8 output boxes"""
    preds = outputs[0][0]  # shape: (84, 8400)
                                                       [ Read 120 lines ]
^G Help         ^O Write Out    ^W Where Is     ^K Cut          ^T Execute      ^C Location     M-U Undo        M-A Set Mark
^X Exit         ^R Read File    ^\ Replace      ^U Paste        ^J Justify      ^/ Go To Line   M-E Redo        M-6 Copy
