
import asyncio
from bleak import BleakServer

SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID = "abcd1234-5678-1234-5678-abcdef123456"

async def main():
    server = BleakServer()
   
    await server.add_service(SERVICE_UUID)
    await server.add_characteristic(
        SERVICE_UUID,
        CHAR_UUID,
        properties=["read", "notify"],
        value=b"chair,0.8,left"
    )

    await server.start()
    print("BLE server running")

    while True:
        await asyncio.sleep(1)

asyncio.run(main())
