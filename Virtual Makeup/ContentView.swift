import SwiftUI

struct ContentView: View {
    @State private var isFaceDetected = false
    @State private var selectedIndex = 0

    private let presets: [(String, UIColor)] = [
        ("Красный", UIColor(red: 0.82, green: 0.10, blue: 0.10, alpha: 1)),
        ("Ягодный", UIColor(red: 0.65, green: 0.12, blue: 0.30, alpha: 1)),
        ("Розовый", UIColor(red: 0.85, green: 0.35, blue: 0.50, alpha: 1)),
        ("Винный", UIColor(red: 0.55, green: 0.08, blue: 0.20, alpha: 1)),
        ("Коралл", UIColor(red: 0.92, green: 0.45, blue: 0.35, alpha: 1)),
        ("Нюд", UIColor(red: 0.72, green: 0.42, blue: 0.35, alpha: 1)),
    ]

    var body: some View {
        ZStack {
            FaceTrackingView(
                isFaceDetected: $isFaceDetected,
                lipColor: presets[selectedIndex].1
            )
            .ignoresSafeArea()

            VStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(isFaceDetected ? .green : .red)
                        .frame(width: 10, height: 10)
                    Text(isFaceDetected ? "Лицо обнаружено" : "Наведите камеру на лицо")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.black.opacity(0.6))
                .clipShape(Capsule())
                .padding(.top, 8)

                Spacer()

                VStack(spacing: 10) {
                    Text("Помада")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)

                    HStack(spacing: 14) {
                        ForEach(presets.indices, id: \.self) { i in
                            Circle()
                                .fill(Color(presets[i].1))
                                .frame(width: 38, height: 38)
                                .overlay(
                                    Circle().strokeBorder(
                                        .white,
                                        lineWidth: selectedIndex == i ? 3 : 0
                                    )
                                )
                                .scaleEffect(selectedIndex == i ? 1.15 : 1.0)
                                .onTapGesture {
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        selectedIndex = i
                                    }
                                }
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 14)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
        }
    }
}
