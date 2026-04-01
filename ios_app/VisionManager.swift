import Foundation
import Vision
import CoreML
import AVFoundation

final class VisionManager {
    static let shared = VisionManager()

    private var lastSpokenTime: Date = .distantPast
    private let cooldown: TimeInterval = 2.0

    private init() {}

    func processFrame(sampleBuffer: CMSampleBuffer) {
        let now = Date()

        guard now.timeIntervalSince(lastSpokenTime) > cooldown else {
            return
        }

        lastSpokenTime = now

        print("Processing frame in VisionManager")

        // Temporary fake detection
        NavigationController().handleObstacle("chair")
    }
}
