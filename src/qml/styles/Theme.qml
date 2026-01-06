pragma Singleton
import QtQuick 2.15

QtObject {
    // =========================================================================
    // Discord Dark Theme Colors
    // =========================================================================

    // Backgrounds
    readonly property color bgPrimary: "#313338"
    readonly property color bgSecondary: "#2b2d31"
    readonly property color bgTertiary: "#1e1f22"
    readonly property color bgModifierHover: "#36373d"
    readonly property color bgModifierSelected: "#404249"
    readonly property color bgModifierActive: "#4e5058"

    // Text
    readonly property color textNormal: "#dbdee1"
    readonly property color textMuted: "#949ba4"
    readonly property color textFaint: "#72767d"
    readonly property color headerPrimary: "#f2f3f5"
    readonly property color headerSecondary: "#b5bac1"

    // Accents
    readonly property color blurple: "#5865f2"
    readonly property color blurpleHover: "#4752c4"
    readonly property color green: "#23a55a"
    readonly property color yellow: "#f0b232"
    readonly property color red: "#da373c"
    readonly property color redHover: "#a12d2f"

    // Interactive
    readonly property color interactiveNormal: "#b5bac1"
    readonly property color interactiveHover: "#dbdee1"
    readonly property color interactiveActive: "#ffffff"

    // Borders
    readonly property color border: "#3f4147"
    readonly property color borderSubtle: "#ffffff0f"

    // =========================================================================
    // Legacy Compatibility (기존 코드와 호환)
    // =========================================================================
    readonly property color accent: blurple
    readonly property color accentHover: blurpleHover
    readonly property color accentCyan: "#00b0b9"
    readonly property color accentPurple: blurple
    readonly property color danger: red
    readonly property color dangerHover: redHover
    readonly property color success: green
    readonly property color warning: yellow
    readonly property color textPrimary: textNormal
    readonly property color textSecondary: textMuted
    readonly property color bgHover: bgModifierHover

    // =========================================================================
    // Typography
    // =========================================================================
    readonly property string fontFamily: "Inter, Pretendard, gg sans, Noto Sans, sans-serif"

    readonly property int fontSizeXSmall: 10
    readonly property int fontSizeSmall: 12
    readonly property int fontSizeMedium: 14
    readonly property int fontSizeLarge: 16
    readonly property int fontSizeXLarge: 20
    readonly property int fontSizeTitle: 24

    // =========================================================================
    // Spacing
    // =========================================================================
    readonly property int spacingXSmall: 4
    readonly property int spacingSmall: 8
    readonly property int spacingMedium: 12
    readonly property int spacingLarge: 16
    readonly property int spacingXLarge: 20
    readonly property int spacingXXLarge: 24

    // =========================================================================
    // Border Radius (Discord는 작은 radius 사용)
    // =========================================================================
    readonly property int radiusSmall: 3
    readonly property int radiusMedium: 4
    readonly property int radiusLarge: 8
    readonly property int radiusXLarge: 12

    // =========================================================================
    // Animation
    // =========================================================================
    readonly property int animFast: 100
    readonly property int animNormal: 200
    readonly property int animSlow: 300

    // =========================================================================
    // Component Heights
    // =========================================================================
    readonly property int buttonHeightSmall: 32
    readonly property int buttonHeightMedium: 38
    readonly property int buttonHeightLarge: 44
    readonly property int inputHeight: 40
    readonly property int sliderHeight: 4
    readonly property int toggleWidth: 40
    readonly property int toggleHeight: 24
}
