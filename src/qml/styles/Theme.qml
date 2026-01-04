pragma Singleton
import QtQuick 2.15

QtObject {
    // Colors (Discord-style based)
    readonly property color bgPrimary: "#1e1f22"
    readonly property color bgSecondary: "#2b2d31"
    readonly property color bgTertiary: "#313338"
    readonly property color bgHover: "#36373d"

    readonly property color accent: "#5865f2"
    readonly property color accentHover: "#4752c4"
    readonly property color accentCyan: "#00D4DB"
    readonly property color accentPurple: "#7B61FF"

    readonly property color textPrimary: "#dbdee1"
    readonly property color textSecondary: "#949ba4"
    readonly property color textMuted: "#72767d"

    readonly property color danger: "#da373c"
    readonly property color dangerHover: "#c93b3e"

    readonly property color success: "#23a55a"
    readonly property color warning: "#f0b232"

    // Fonts
    readonly property string fontFamily: "Inter, Pretendard, Segoe UI, sans-serif"
    readonly property int fontSizeSmall: 11
    readonly property int fontSizeMedium: 13
    readonly property int fontSizeLarge: 16
    readonly property int fontSizeTitle: 20

    // Spacing
    readonly property int spacingSmall: 8
    readonly property int spacingMedium: 12
    readonly property int spacingLarge: 16
    readonly property int spacingXLarge: 24

    // Border radius
    readonly property int radiusSmall: 4
    readonly property int radiusMedium: 8
    readonly property int radiusLarge: 12
    readonly property int radiusXLarge: 16

    // Animation durations (ms)
    readonly property int animFast: 150
    readonly property int animNormal: 250
    readonly property int animSlow: 400

    // Shadows
    readonly property string shadowSmall: "0 1px 2px rgba(0, 0, 0, 0.3)"
    readonly property string shadowMedium: "0 4px 12px rgba(0, 0, 0, 0.4)"
    readonly property string shadowLarge: "0 8px 24px rgba(0, 0, 0, 0.5)"
}
