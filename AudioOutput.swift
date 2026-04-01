import Foundation
import AVFoundation

final class AudioOutput {
    static let shared = AudioOutput()

    private let synthesizer = AVSpeechSynthesizer()

    private init() {}

    func speak(_ text: String) {
        guard !text.isEmpty else { return }

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.5
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        synthesizer.speak(utterance)
    }
}
