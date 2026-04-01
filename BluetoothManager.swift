import Foundation
import CoreBluetooth

final class BluetoothManager: NSObject, CBCentralManagerDelegate {
    private var centralManager: CBCentralManager?

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("Bluetooth is powered on")
        case .poweredOff:
            print("Bluetooth is powered off")
        case .unauthorized:
            print("Bluetooth unauthorized")
        case .unsupported:
            print("Bluetooth unsupported")
        case .resetting:
            print("Bluetooth resetting")
        case .unknown:
            print("Bluetooth state unknown")
        @unknown default:
            print("Unknown Bluetooth state")
        }
    }
}
