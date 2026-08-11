import SwiftUI

struct ContentView: View {
    private enum MakeupTool: String, CaseIterable, Identifiable {
        case lipstick = "Помада"
        case blush = "Румяна"

        var id: String { rawValue }
    }

    private struct LipPreset {
        let name: String
        let color: UIColor
        let finish: LipFinish
    }

    private struct BlushPreset {
        let name: String
        let color: UIColor
    }

    @State private var isFaceDetected = false
    @State private var selectedTool: MakeupTool = .lipstick
    @State private var selectedIndex = 0
    @State private var selectedBlushIndex = 0
    // Real lipstick never fully replaces the captured lip colour. Start with
    // a translucent layer and keep a little camera detail even at the upper
    // end of the user control.
    @State private var lipstickOpacity = 1.0
    @State private var blushOpacity = 0.55
    @State private var isShowingOriginal = false

    private let presets: [LipPreset] = [
        LipPreset(
            name: "131, Guerlain",
            color: UIColor(
                red: 194.0 / 255.0,
                green: 118.0 / 255.0,
                blue: 105.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),
        LipPreset(
            name: "518, Guerlain",
            color: UIColor(
                red: 209.0 / 255.0,
                green: 107.0 / 255.0,
                blue: 106.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),
        LipPreset(
            name: "06, Guerlain",
            color: UIColor(
                red: 176.0 / 255.0,
                green: 85.0 / 255.0,
                blue: 85.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),
        LipPreset(
            name: "Clarins, 705S",
            color: UIColor(
                red: 151.0 / 255.0,
                green: 75.0 / 255.0,
                blue: 72.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),
        LipPreset(
            name: "770, Guerlain",
            color: UIColor(
                red: 172.0 / 255.0,
                green: 41.0 / 255.0,
                blue: 54.0 / 255.0,
                alpha: 1.0
            ),
            finish: .matte
        ),
        LipPreset(
            name: "360, Guerlain",
            color: UIColor(
                red: 163.0 / 255.0,
                green: 91.0 / 255.0,
                blue: 76.0 / 255.0,
                alpha: 1.0
            ),
            finish: .matte
        ),
        LipPreset(
            name: "Burning Love, MAC",
            color: UIColor(
                red: 116.0 / 255.0,
                green: 30.0 / 255.0,
                blue: 50.0 / 255.0,
                alpha: 1.0
            ),
            finish: .matte
        ),
        LipPreset(
            name: "Brave, MAC",
            color: UIColor(
                red: 180.0 / 255.0,
                green: 66.0 / 255.0,
                blue: 85.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),
        
        LipPreset(
            name: "999, DIOR",
            color: UIColor(
                red: 158.0 / 255.0,
                green: 38.0 / 255.0,
                blue: 32.0 / 255.0,
                alpha: 1.0
            ),
            finish: .satin
        ),

        LipPreset(
            name: "Marrakeshmere, MAC",
            color: UIColor(
                red: 120.0 / 255.0,
                green: 44.0 / 255.0,
                blue: 36.0 / 255.0,
                alpha: 1.0
            ),
            finish: .matte
        ),
    ]

    private let blushPresets: [BlushPreset] = [
        BlushPreset(name: "Нежный розовый", color: UIColor(red: 0.95, green: 0.42, blue: 0.50, alpha: 1)),
        BlushPreset(name: "Персиковый", color: UIColor(red: 0.96, green: 0.50, blue: 0.38, alpha: 1)),
        BlushPreset(name: "Пыльная роза", color: UIColor(red: 0.75, green: 0.34, blue: 0.40, alpha: 1)),
        BlushPreset(name: "Коралловый", color: UIColor(red: 0.93, green: 0.36, blue: 0.32, alpha: 1))
    ]

    var body: some View {
        ZStack {
            FaceTrackingView(
                isFaceDetected: $isFaceDetected,
                lipColor: presets[selectedIndex].color,
                lipOpacity: isShowingOriginal ? 0 : lipstickOpacity,
                lipFinish: presets[selectedIndex].finish,
                blushColor: blushPresets[selectedBlushIndex].color,
                blushOpacity: isShowingOriginal ? 0 : blushOpacity
            )
            .ignoresSafeArea()

            VStack {
                HStack(spacing: 10) {
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

                    Spacer(minLength: 0)

                    Button {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            isShowingOriginal.toggle()
                        }
                    } label: {
                        Label(
                            "Оригинал",
                            systemImage: isShowingOriginal ? "eye.fill" : "eye"
                        )
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(isShowingOriginal ? .black : .white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(
                            isShowingOriginal ?
                                Color.white : Color.black.opacity(0.6)
                        )
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(
                        isShowingOriginal ?
                            "Вернуть виртуальный макияж" :
                            "Показать оригинал без макияжа"
                    )
                    .accessibilityAddTraits(isShowingOriginal ? .isSelected : [])
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                Spacer()

                VStack(spacing: 12) {
                    Picker("Макияж", selection: $selectedTool) {
                        ForEach(MakeupTool.allCases) { tool in
                            Text(tool.rawValue).tag(tool)
                        }
                    }
                    .pickerStyle(.segmented)

                    HStack {
                        Text(selectedTool.rawValue)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                        Spacer()
                        Text("\(Int(activeOpacity * 100))%")
                            .font(.caption2.monospacedDigit().weight(.semibold))
                            .foregroundStyle(.white.opacity(0.86))
                    }
                    .padding(.horizontal, 4)

                    Slider(value: activeOpacityBinding, in: activeOpacityRange)
                        .tint(.white)
                        .frame(maxWidth: 360)
                        .accessibilityLabel("Интенсивность \(selectedTool.rawValue.lowercased())")

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 14) {
                            if selectedTool == .lipstick {
                                ForEach(presets.indices, id: \.self) { i in
                                    VStack(spacing: 7) {
                                        Circle()
                                            .fill(Color(presets[i].color))
                                            .frame(width: 38, height: 38)
                                            .overlay(alignment: .topLeading) {
                                                if presets[i].finish == .gloss {
                                                    Capsule()
                                                        .fill(.white.opacity(0.68))
                                                        .frame(width: 17, height: 6)
                                                        .rotationEffect(.degrees(-24))
                                                        .offset(x: 8, y: 8)
                                                }
                                            }
                                            .overlay(
                                                Circle().strokeBorder(
                                                    .white,
                                                    lineWidth: selectedIndex == i ? 3 : 0
                                                )
                                            )
                                            .scaleEffect(selectedIndex == i ? 1.12 : 1.0)

                                        Text(presets[i].name)
                                            .font(.caption2.weight(selectedIndex == i ? .semibold : .regular))
                                            .foregroundStyle(.white.opacity(selectedIndex == i ? 1 : 0.78))
                                            .multilineTextAlignment(.center)
                                            .lineLimit(2)
                                            .minimumScaleFactor(0.72)
                                            .frame(width: 82, height: 28, alignment: .top)
                                    }
                                    .contentShape(Rectangle())
                                    .accessibilityElement(children: .ignore)
                                    .accessibilityLabel(presets[i].name)
                                    .accessibilityAddTraits(selectedIndex == i ? [.isButton, .isSelected] : .isButton)
                                    .onTapGesture {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            selectedIndex = i
                                        }
                                    }
                                }
                            } else {
                                ForEach(blushPresets.indices, id: \.self) { i in
                                    VStack(spacing: 7) {
                                        Circle()
                                            .fill(Color(blushPresets[i].color))
                                            .frame(width: 38, height: 38)
                                            .overlay(
                                                Circle().strokeBorder(
                                                    .white,
                                                    lineWidth: selectedBlushIndex == i ? 3 : 0
                                                )
                                            )
                                            .scaleEffect(selectedBlushIndex == i ? 1.12 : 1.0)

                                        Text(blushPresets[i].name)
                                            .font(.caption2.weight(selectedBlushIndex == i ? .semibold : .regular))
                                            .foregroundStyle(.white.opacity(selectedBlushIndex == i ? 1 : 0.78))
                                            .multilineTextAlignment(.center)
                                            .lineLimit(2)
                                            .minimumScaleFactor(0.72)
                                            .frame(width: 82, height: 28, alignment: .top)
                                    }
                                    .contentShape(Rectangle())
                                    .accessibilityElement(children: .ignore)
                                    .accessibilityLabel(blushPresets[i].name)
                                    .accessibilityAddTraits(selectedBlushIndex == i ? [.isButton, .isSelected] : .isButton)
                                    .onTapGesture {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            selectedBlushIndex = i
                                        }
                                    }
                                }
                            }
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 3)
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

    private var activeOpacity: Double {
        selectedTool == .lipstick ? lipstickOpacity : blushOpacity
    }

    private var activeOpacityBinding: Binding<Double> {
        selectedTool == .lipstick ? $lipstickOpacity : $blushOpacity
    }

    private var activeOpacityRange: ClosedRange<Double> {
        selectedTool == .lipstick ? 0.35...1 : 0.0...0.85
    }
}
