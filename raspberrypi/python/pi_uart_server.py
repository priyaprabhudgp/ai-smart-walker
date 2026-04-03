from bluezero import peripheral
from bluezero.adapter import Adapter
import threading
import time

SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID = "abcd1234-5678-1234-5678-abcdef123456"

current_value = bytearray(b"chair,0.8,left")

def read_value():
    return list(current_value)

def notify_loop(dongle, chrc):
    global current_value
    samples = [
        b"chair,0.8,left",
        b"person,1.2,center",
        b"table,0.6,right"
    ]
    i = 0
    while True:
        current_value = bytearray(samples[i % len(samples)])
        chrc.set_value(list(current_value))
        chrc.notify_value()
        print("Sent:", current_value.decode())
        i += 1
        time.sleep(3)

adapter_address = list(Adapter.available())[0].address

walker = peripheral.Peripheral(
    adapter_address=adapter_address,
    local_name="SmartWalkerPi",
    appearance=0
)

walker.add_service(srv_id=1, uuid=SERVICE_UUID, primary=True)
walker.add_characteristic(
    srv_id=1,
    chr_id=1,
    uuid=CHAR_UUID,
    value=[],
    notifying=True,
    flags=['read', 'notify'],
    read_callback=read_value
)

def main():
    walker.publish()
    chrc = walker.characteristics[0]
    thread = threading.Thread(target=notify_loop, args=(walker, chrc), daemon=True)
    thread.start()
    print("SmartWalkerPi BLE peripheral running")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        walker.unpublish()

if __name__ == "__main__":
    main()
