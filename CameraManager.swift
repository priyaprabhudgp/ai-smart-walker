import Foundation
import AVFoundation
import Combine

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    private var isConfigured = false

    func startSession() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        switch status {
        case .authorized:
            setupAndStartSession()

        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.setupAndStartSession()
                    } else {
                        print("Camera permission denied")
                    }
                }
            }

        case .denied, .restricted:
            print("Camera access denied or restricted")

        @unknown default:
            print("Unknown camera authorization status")
        }
    }

    private func setupAndStartSession() {
        if session.isRunning {
            print("Camera already running")
            return
        }

        if !isConfigured {
            session.beginConfiguration()
            session.sessionPreset = .medium

            guard let device = AVCaptureDevice.default(for: .video) else {
                print("No video camera available on this device")
                session.commitConfiguration()
                return
            }

            guard let input = try? AVCaptureDeviceInput(device: device) else {
                print("Could not create camera input")
                session.commitConfiguration()
                return
            }

            if session.canAddInput(input) {
                session.addInput(input)
            }

            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame"))

            if session.canAddOutput(output) {
                session.addOutput(output)
            }

            session.commitConfiguration()
            isConfigured = true
        }

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
            print("Camera started")
        }
    }

    func stopSession() {
        guard session.isRunning else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.stopRunning()
            print("Camera stopped")
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        VisionManager.shared.processFrame(sampleBuffer: sampleBuffer)
    }
}
