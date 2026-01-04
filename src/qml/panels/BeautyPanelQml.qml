import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../styles"
import "../components"

Rectangle {
    id: root
    color: "#0A0A0A"

    // Header
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 48
        color: "#101010"

        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: 1
            color: Qt.rgba(1, 1, 1, 0.06)
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16

            Text {
                text: "MUSE"
                color: Theme.accentCyan
                font.pixelSize: 12
                font.weight: Font.Bold
                font.letterSpacing: 2
            }

            Item { Layout.fillWidth: true }

            Text {
                text: beautyBridge.profileName.toUpperCase()
                color: Qt.rgba(1, 1, 1, 0.4)
                font.pixelSize: 11
                font.weight: Font.Normal
            }
        }
    }

    // Scrollable Content
    ScrollView {
        id: scrollView
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            width: scrollView.width
            spacing: 20

            Item { Layout.preferredHeight: 10 }

            // =====================================================================
            // Background Required Warning
            // =====================================================================
            Rectangle {
                id: bgRequiredFrame
                visible: !beautyBridge.hasBackground
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15
                Layout.preferredHeight: bgContent.implicitHeight + 36
                color: Qt.rgba(1, 0.32, 0.32, 0.06)
                border.color: Qt.rgba(1, 0.32, 0.32, 0.25)
                border.width: 1
                radius: 14

                Column {
                    id: bgContent
                    anchors.fill: parent
                    anchors.margins: 18
                    spacing: 10

                    Text {
                        text: "\u26A0\uFE0F \uBC30\uACBD \uC800\uC7A5 \uD544\uC694"
                        color: "#FF5252"
                        font.pixelSize: 13
                        font.weight: Font.DemiBold
                        font.letterSpacing: 0.3
                    }

                    Text {
                        text: "\uBC30\uACBD\uC744 \uC800\uC7A5\uD574\uC57C \uBCF4\uC815\uC774 \uD65C\uC131\uD654\uB429\uB2C8\uB2E4.\n\uCE74\uBA54\uB77C \uD654\uBA74\uC5D0\uC11C \uBC97\uC5B4\uB09C \uCC44\uB85C \uBC30\uACBD\uC744 \uC800\uC7A5\uD574\uC8FC\uC138\uC694."
                        color: Qt.rgba(1, 1, 1, 0.5)
                        font.pixelSize: 12
                        lineHeight: 1.5
                        wrapMode: Text.WordWrap
                        width: parent.width
                    }

                    ModernButton {
                        text: "\uD83D\uDCF7 \uBC30\uACBD \uC800\uC7A5\uD558\uAE30 (\uB2E8\uCD95\uD0A4: B)"
                        buttonType: "gradient"
                        width: parent.width
                        height: 44
                        onClicked: beautyBridge.captureBackground()
                    }
                }
            }

            // =====================================================================
            // Group 1: Face Shape
            // =====================================================================
            GroupBox {
                title: "\uC5BC\uAD74 \uC724\uACFD (FACE SHAPE)"
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15
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
                    label: "\uCF67\uBCFC \uC870\uC808"
                    value: beautyBridge.noseSlim
                    Layout.fillWidth: true
                    onSliderMoved: beautyBridge.noseSlim = newValue
                }
            }

            // =====================================================================
            // Group 2: Body Shape
            // =====================================================================
            GroupBox {
                title: "\uCCB4\uD615 \uBCF4\uC815 (BODY SHAPE)"
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15
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
                title: "\uD53C\uBD80 \uAE30\uBCF8 (SKIN BASIC)"
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15

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
                title: "\uC0C9\uC0C1 \uC870\uC815 (COLOR GRADING)"
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15

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
                title: "\uC124\uC815 (SETTINGS)"
                Layout.fillWidth: true
                Layout.leftMargin: 15
                Layout.rightMargin: 15

                ModernCheckBox {
                    text: "AI \uAD00\uC808 / \uB9C8\uC2A4\uD06C \uBCF4\uAE30"
                    checked: beautyBridge.showBodyDebug
                    onCheckedChanged: beautyBridge.showBodyDebug = checked
                }
            }

            // Bottom spacer
            Item { Layout.preferredHeight: 20 }
        }
    }
}
