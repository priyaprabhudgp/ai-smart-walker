import Foundation

final class NavigationController {
    func handleObstacle(_ name: String) {
        switch name.lowercased() {
        case "chair":
            AudioOutput.shared.speak("Chair ahead")
        case "person":
            AudioOutput.shared.speak("Person ahead")
        case "table":
            AudioOutput.shared.speak("Table ahead")
        default:
            AudioOutput.shared.speak("Obstacle ahead")
        }
    }
}
