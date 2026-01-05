import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../styles"
import "../components"

Rectangle {
    id: root
    color: Theme.bgSecondary

    // Scrollable Content
    ScrollView {
        id: scrollView
        anchors.fill: parent
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ScrollBar.vertical.policy: ScrollBar.AsNeeded

        ColumnLayout {
            width: scrollView.width
            spacing: Theme.spacingXLarge

            // Top spacer
            Item { Layout.preferredHeight: Theme.spacingLarge }

            // =====================================================================
            // Background Required Warning
            // =====================================================================
            Rectangle {
                id: bgRequiredFrame
                visible: !beautyBridge.hasBackground
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge
                Layout.preferredHeight: bgContent.implicitHeight + Theme.spacingXLarge
                color: Qt.rgba(Theme.yellow.r, Theme.yellow.g, Theme.yellow.b, 0.1)
                border.color: Qt.rgba(Theme.yellow.r, Theme.yellow.g, Theme.yellow.b, 0.3)
                border.width: 1
                radius: Theme.radiusLarge

                Column {
                    id: bgContent
                    anchors.fill: parent
                    anchors.margins: Theme.spacingMedium
                    spacing: Theme.spacingSmall

                    Row {
                        spacing: Theme.spacingSmall

                        Text {
                            text: "\u26A0"
                            color: Theme.yellow
                            font.pixelSize: Theme.fontSizeMedium
                        }

                        Text {
                            text: "\uBC30\uACBD \uC800\uC7A5 \uD544\uC694"
                            color: Theme.yellow
                            font.pixelSize: Theme.fontSizeMedium
                            font.weight: Font.DemiBold
                        }
                    }

                    Text {
                        text: "\uBC30\uACBD\uC744 \uC800\uC7A5\uD574\uC57C \uBCF4\uC815\uC774 \uD65C\uC131\uD654\uB429\uB2C8\uB2E4.\n\uCE74\uBA54\uB77C \uD654\uBA74\uC5D0\uC11C \uBC97\uC5B4\uB09C \uCC44\uB85C \uBC30\uACBD\uC744 \uC800\uC7A5\uD574\uC8FC\uC138\uC694."
                        color: Theme.textMuted
                        font.pixelSize: Theme.fontSizeSmall
                        lineHeight: 1.4
                        wrapMode: Text.WordWrap
                        width: parent.width
                    }

                    ModernButton {
                        text: "\uBC30\uACBD \uC800\uC7A5\uD558\uAE30 (B)"
                        buttonType: "primary"
                        width: parent.width
                        height: Theme.buttonHeightMedium
                        onClicked: beautyBridge.captureBackground()
                    }
                }
            }

            // =====================================================================
            // Group 1: Face Shape
            // =====================================================================
            GroupBox {
                title: "\uC5BC\uAD74 \uC724\uACFD"
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge
                enabled: beautyBridge.hasBackground
                opacity: enabled ? 1.0 : 0.5

                Behavior on opacity {
                    NumberAnimation { duration: Theme.animNormal }
                }

                ModernSlider {
                    label: "\uD131 \uAE4E\uAE30 (V-Line)"
                    value: beautyBridge.faceV
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.faceV = newValue
                }

                ModernSlider {
                    label: "\uB208 \uD06C\uAE30 \uC870\uC808"
                    value: beautyBridge.eyeScale
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.eyeScale = newValue
                }

                ModernSlider {
                    label: "\uCF67\uBCBC \uC870\uC808"
                    value: beautyBridge.noseSlim
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.noseSlim = newValue
                }
            }

            // =====================================================================
            // Group 2: Body Shape
            // =====================================================================
            GroupBox {
                title: "\uCCB4\uD615 \uBCF4\uC815"
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge
                enabled: beautyBridge.hasBackground
                opacity: enabled ? 1.0 : 0.5

                Behavior on opacity {
                    NumberAnimation { duration: Theme.animNormal }
                }

                ModernSlider {
                    label: "\uD5C8\uB9AC \uC904\uC774\uAE30"
                    value: beautyBridge.waistSlim
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.waistSlim = newValue
                }

                ModernSlider {
                    label: "\uACE8\uBC18 \uB298\uB9AC\uAE30"
                    value: beautyBridge.hipWiden
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.hipWiden = newValue
                }
            }

            // =====================================================================
            // Group 3: Skin Basic
            // =====================================================================
            GroupBox {
                title: "\uD53C\uBD80 \uAE30\uBCF8"
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge

                ModernSlider {
                    label: "\uD53C\uBD80 \uACB0 \uBCF4\uC815"
                    value: beautyBridge.skinSmooth
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.skinSmooth = newValue
                }

                ModernSlider {
                    label: "\uCE58\uC544 \uBBF8\uBC31"
                    value: beautyBridge.teethWhiten
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.teethWhiten = newValue
                }
            }

            // =====================================================================
            // Group 4: Color Grading
            // =====================================================================
            GroupBox {
                title: "\uC0C9\uC0C1 \uC870\uC815"
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge

                ModernSlider {
                    label: "\uC0C9\uC628\uB3C4 (Cool \u2194 Warm)"
                    value: beautyBridge.colorTemperature
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.colorTemperature = newValue
                }
            }

            // =====================================================================
            // Group 5: Settings
            // =====================================================================
            GroupBox {
                title: "\uC124\uC815"
                Layout.fillWidth: true
                Layout.leftMargin: Theme.spacingLarge
                Layout.rightMargin: Theme.spacingLarge

                ModernCheckBox {
                    text: "AI \uAD00\uC808 / \uB9C8\uC2A4\uD06C \uBCF4\uAE30"
                    checked: beautyBridge.showBodyDebug
                    onToggled: beautyBridge.showBodyDebug = newValue
                }
            }

            // Bottom spacer
            Item { Layout.preferredHeight: Theme.spacingLarge }
        }
    }
}
