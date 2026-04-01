import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()

    var body: some View {
        VStack(spacing: 16) {
            Text("Smart Walker")
                .font(.largeTitle)
                .bold()

            ZStack {
                CameraPreviewView(session: cameraManager.session)
                    .frame(height: 400)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                // Fake debug box for now
                Rectangle()
                    .stroke(lineWidth: 3)
                    .frame(width: 140, height: 120)
            }

            HStack(spacing: 20) {
                Button("Start Camera") {
                    cameraManager.startSession()
                }

                Button("Stop Camera") {
                    cameraManager.stopSession()
                }

                Button("Test Alert") {
                    NavigationController().handleObstacle("chair")
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
