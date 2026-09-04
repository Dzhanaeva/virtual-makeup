import SwiftUI

struct ContentView: View {
    private enum MakeupTool: String, CaseIterable, Identifiable {
        case lipstick = "Помада"
        case blush = "Румяна"

        var id: String { rawValue }
    }

    private struct LipShade: Identifiable {
        /// Temporary catalogue identifier for the shade/SKU.
        let productID: String
        let name: String
        let color: LipstickColor

        var id: String { productID }
    }

    private struct LipProduct: Identifiable {
        let id: String
        let placeholderNumber: Int
        let name: String
        let productType: LipProductType
        let finish: LipFinish
        let density: LipstickDensity
        let texture: LipstickTexture
        let shades: [LipShade]

        init(
            id: String,
            placeholderNumber: Int,
            name: String,
            productType: LipProductType = .lipstick,
            finish: LipFinish,
            density: LipstickDensity,
            texture: LipstickTexture,
            shades: [LipShade]
        ) {
            self.id = id
            self.placeholderNumber = placeholderNumber
            self.name = name
            self.productType = productType
            self.finish = finish
            self.density = density
            self.texture = texture
            self.shades = shades
        }
    }

    private struct BlushPreset {
        let name: String
        let color: UIColor
    }

    @State private var isFaceDetected = false
    @State private var selectedTool: MakeupTool = .lipstick
    @State private var selectedProductIndex = 0
    @State private var selectedShadeIndex = 0
    @State private var isShowingLipProductDetails = false
    @State private var selectedBlushIndex = 0
    // Real lipstick never fully replaces the captured lip colour. Start with
    // a translucent layer and keep a little camera detail even at the upper
    // end of the user control.
    @State private var lipstickOpacity = 1.0
    @State private var blushOpacity = 0.55
    @State private var isShowingOriginal = false

    private let lipProducts: [LipProduct] = [
        LipProduct(
            id: "guerlain-rouge-g-satin",
            placeholderNumber: 1,
            name: "Guerlain Rouge G Satin",
            finish: .satin,
            density: .medium,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1001-131",
                    name: "131",
                    color: LipstickColor(sourceSRGBHex: 0xC3776A)
                ),
                LipShade(
                    productID: "VM-1001-518",
                    name: "518",
                    color: LipstickColor(sourceSRGBHex: 0xD26B6B)
                ),
                LipShade(
                    productID: "VM-1001-006",
                    name: "06",
                    color: LipstickColor(sourceSRGBHex: 0xAF5655)
                )
            ]
        ),
        LipProduct(
            id: "dior-rouge-dior-satin",
            placeholderNumber: 2,
            name: "Dior Rouge Dior Satin",
            finish: .satin,
            density: .high,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1002-999",
                    name: "999",
                    color: LipstickColor(sourceSRGBHex: 0xB8202D)
                )
            ]
        ),
        LipProduct(
            id: "guerlain-rouge-g-velvet",
            placeholderNumber: 3,
            name: "Guerlain Rouge G Velvet",
            finish: .matte,
            density: .high,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1003-770",
                    name: "770",
                    color: LipstickColor(sourceSRGBHex: 0xAC2936)
                ),
                LipShade(
                    productID: "VM-1003-360",
                    name: "360",
                    color: LipstickColor(sourceSRGBHex: 0xA35B4C)
                )
            ]
        ),
        LipProduct(
            id: "mac-powder-kiss",
            placeholderNumber: 4,
            name: "MAC Powder Kiss",
            finish: .matte,
            density: .high,
            texture: .mousse,
            shades: [
                LipShade(
                    productID: "VM-1004-BL",
                    name: "Burning Love",
                    color: LipstickColor(sourceSRGBHex: 0x7E1331)
                ),
                LipShade(
                    productID: "VM-1004-MM",
                    name: "Marrakeshmere",
                    color: LipstickColor(sourceSRGBHex: 0x782C24)
                )
            ]
        ),
        LipProduct(
            id: "mac-satin-lipstick",
            placeholderNumber: 5,
            name: "MAC Satin Lipstick",
            finish: .satin,
            density: .medium,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1005-BRV",
                    name: "Brave",
                    color: LipstickColor(sourceSRGBHex: 0xB44255)
                )
            ]
        ),
        LipProduct(
            id: "clarins-satin-lipstick",
            placeholderNumber: 6,
            name: "Clarins Satin Lipstick",
            finish: .satin,
            density: .medium,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1006-705S",
                    name: "705S",
                    color: LipstickColor(sourceSRGBHex: 0xA34545)
                )
            ]
        ),
        LipProduct(
            id: "clarins-lip-gloss",
            placeholderNumber: 7,
            name: "Clarins Lip Gloss",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1007-001",
                    name: "01",
                    color: LipstickColor(sourceSRGBHex: 0xE39294)
                )
            ]
        ),
        
        LipProduct(
            id: "dior-lip-gloss",
            placeholderNumber: 8,
            name: "Dior Lip Gloss Addict",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1008-006",
                    name: "006",
                    color: LipstickColor(sourceSRGBHex: 0xA74C89)
                ),
                LipShade(
                    productID: "VM-1008-012",
                    name: "012",
                    color: LipstickColor(sourceSRGBHex: 0xBC6065)
                )
            ]
        ),
        
        LipProduct(
            id: "dior-lip-gloss-creamy",
            placeholderNumber: 9,
            name: "Dior Lip Gloss",
            finish: .gloss,
            density: .low,
            texture: .creamy,
            shades: [
                LipShade(
                    productID: "VM-1008-040",
                    name: "040",
                    color: LipstickColor(sourceSRGBHex: 0xA52D3C)
                ),
                LipShade(
                    productID: "VM-1008-004",
                    name: "004",
                    color: LipstickColor(sourceSRGBHex: 0xEB8571)
                )
            ]
        ),
        
        LipProduct(
            id: "shik-lip-gloss",
            placeholderNumber: 10,
            name: "Shik Lip Gloss Care",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1008-008",
                    name: "08",
                    color: LipstickColor(sourceSRGBHex: 0x371A12)
                )
            ]
        ),
        
        LipProduct(
            id: "make-up-forever-lip-gloss",
            placeholderNumber: 11,
            name: "Make Up Forever Lip Gloss",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1008-006",
                    name: "06",
                    color: LipstickColor(sourceSRGBHex: 0xBE7C6E)
                )
            ]
        ),
        
        LipProduct(
            id: "shik-lip-gloss-balm",
            placeholderNumber: 12,
            name: "Shik Total repair Balm",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1008-001",
                    name: "01",
                    color: LipstickColor(sourceSRGBHex: 0xF7C8BE)
                ),
                LipShade(
                    productID: "VM-1008-002",
                    name: "02",
                    color: LipstickColor(sourceSRGBHex: 0xAF584F)
                ),
                LipShade(
                    productID: "VM-1008-005",
                    name: "05",
                    color: LipstickColor(sourceSRGBHex: 0x2E0201)
                ),
            ]
        ),
        
        LipProduct(
            id: "love-generation-lip-gloss",
            placeholderNumber: 11,
            name: "Love Generation Lip Gloss Wet Dream",
            finish: .gloss,
            density: .low,
            texture: .liquid,
            shades: [
                LipShade(
                    productID: "VM-1009-006",
                    name: "06",
                    color: LipstickColor(sourceSRGBHex: 0xB0453D)
                )
            ]
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
                lipColor: selectedLipShade.color,
                lipOpacity: isShowingOriginal ? 0 : lipstickOpacity,
                lipFinish: selectedLipProduct.finish,
                lipDensity: selectedLipProduct.density,
                lipTexture: selectedLipProduct.texture,
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
                        Text(activeToolTitle)
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

                    if selectedTool == .lipstick {
                        lipstickProductAndShadePicker
                    } else {
                        blushPicker
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

    @ViewBuilder
    private var lipstickProductAndShadePicker: some View {
        if isShowingLipProductDetails {
            selectedLipProductPicker
                .transition(.opacity.combined(with: .move(edge: .trailing)))
        } else {
            lipProductCatalog
                .transition(.opacity.combined(with: .move(edge: .leading)))
        }
    }

    private var lipProductCatalog: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Продукты для губ")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.72))
                .padding(.horizontal, 4)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ForEach(lipProducts.indices, id: \.self) { index in
                        let product = lipProducts[index]
                        Button {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                selectedProductIndex = index
                                selectedShadeIndex = 0
                                isShowingLipProductDetails = true
                            }
                        } label: {
                            lipProductTile(
                                product,
                                isSelected: selectedProductIndex == index
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(
                            "\(product.productType.displayName) \(product.name), \(product.finish.displayName), плотность \(product.density.displayName.lowercased()), текстура \(product.texture.displayName.lowercased())"
                        )
                        .accessibilityAddTraits(
                            selectedProductIndex == index ? .isSelected : []
                        )
                    }
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 3)
            }
        }
    }

    private var selectedLipProductPicker: some View {
        HStack(alignment: .top, spacing: 10) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isShowingLipProductDetails = false
                }
            } label: {
                Image(systemName: "chevron.left")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 30, height: 30)
                    .background(.black.opacity(0.32))
                    .clipShape(Circle())
                    .overlay(
                        Circle().strokeBorder(.white.opacity(0.50), lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Вернуться к списку продуктов для губ")
            .padding(.top, 7)

            lipProductTile(selectedLipProduct, isSelected: true)

            Rectangle()
                .fill(.white.opacity(0.20))
                .frame(width: 1, height: 76)
                .padding(.horizontal, 2)

            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text("Оттенки")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.72))
                    Spacer(minLength: 8)
                    Text("ID: \(selectedLipShade.productID)")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.62))
                        .lineLimit(1)
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(selectedLipProduct.shades.indices, id: \.self) { index in
                            let shade = selectedLipProduct.shades[index]
                            Button {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    selectedShadeIndex = index
                                }
                            } label: {
                                VStack(spacing: 5) {
                                    Circle()
                                        .fill(Color(shade.color.uiColor))
                                        .frame(width: 40, height: 40)
                                        .overlay(alignment: .topLeading) {
                                            if selectedLipProduct.finish == .gloss {
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
                                                lineWidth: selectedShadeIndex == index ? 3 : 0
                                            )
                                        )
                                        .scaleEffect(selectedShadeIndex == index ? 1.08 : 1)

                                    Text(shade.name)
                                        .font(.caption2.weight(
                                            selectedShadeIndex == index ? .semibold : .regular
                                        ))
                                        .foregroundStyle(.white.opacity(
                                            selectedShadeIndex == index ? 1 : 0.78
                                        ))
                                        .lineLimit(1)

                                    Text(shade.productID)
                                        .font(.system(size: 8, design: .monospaced))
                                        .foregroundStyle(.white.opacity(0.56))
                                        .lineLimit(1)
                                }
                                .frame(width: 88)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(
                                "Оттенок \(shade.name), ID товара \(shade.productID)"
                            )
                            .accessibilityAddTraits(
                                selectedShadeIndex == index ? .isSelected : []
                            )
                        }
                    }
                    .padding(.horizontal, 3)
                    .padding(.vertical, 3)
                }
            }
        }
    }

    private func lipProductTile(_ product: LipProduct,
                                isSelected: Bool) -> some View {
        VStack(spacing: 6) {
            Circle()
                .fill(isSelected ? Color.white : Color.black.opacity(0.32))
                .frame(width: 44, height: 44)
                .overlay {
                    Text("\(product.placeholderNumber)")
                        .font(.headline.monospacedDigit().weight(.bold))
                        .foregroundStyle(isSelected ? .black : .white)
                }
                .overlay(
                    Circle().strokeBorder(
                        .white.opacity(0.88),
                        lineWidth: isSelected ? 3 : 1
                    )
                )

            Text(product.name)
                .font(.caption2.weight(isSelected ? .semibold : .regular))
                .foregroundStyle(.white.opacity(isSelected ? 1 : 0.76))
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .minimumScaleFactor(0.70)
                .frame(width: 96, height: 28, alignment: .top)
        }
    }

    private var blushPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 14) {
                ForEach(blushPresets.indices, id: \.self) { index in
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedBlushIndex = index
                        }
                    } label: {
                        VStack(spacing: 7) {
                            Circle()
                                .fill(Color(blushPresets[index].color))
                                .frame(width: 38, height: 38)
                                .overlay(
                                    Circle().strokeBorder(
                                        .white,
                                        lineWidth: selectedBlushIndex == index ? 3 : 0
                                    )
                                )
                                .scaleEffect(selectedBlushIndex == index ? 1.12 : 1)

                            Text(blushPresets[index].name)
                                .font(.caption2.weight(
                                    selectedBlushIndex == index ? .semibold : .regular
                                ))
                                .foregroundStyle(.white.opacity(
                                    selectedBlushIndex == index ? 1 : 0.78
                                ))
                                .multilineTextAlignment(.center)
                                .lineLimit(2)
                                .minimumScaleFactor(0.72)
                                .frame(width: 82, height: 28, alignment: .top)
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(blushPresets[index].name)
                    .accessibilityAddTraits(
                        selectedBlushIndex == index ? .isSelected : []
                    )
                }
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 3)
        }
    }

    private var selectedLipProduct: LipProduct {
        lipProducts[selectedProductIndex]
    }

    private var selectedLipShade: LipShade {
        selectedLipProduct.shades[selectedShadeIndex]
    }

    private var activeOpacity: Double {
        selectedTool == .lipstick ? lipstickOpacity : blushOpacity
    }

    private var activeToolTitle: String {
        guard selectedTool == .lipstick else {
            return selectedTool.rawValue
        }
        return "\(selectedLipProduct.productType.displayName) · \(selectedLipProduct.finish.displayName)"
    }

    private var activeOpacityBinding: Binding<Double> {
        selectedTool == .lipstick ? $lipstickOpacity : $blushOpacity
    }

    private var activeOpacityRange: ClosedRange<Double> {
        selectedTool == .lipstick ? 0.35...1 : 0.0...0.85
    }
}
