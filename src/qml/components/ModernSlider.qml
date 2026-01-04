import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../styles"

Item {
    id: root
    implicitHeight: 52
    implicitWidth: 280

    // Properties
    property string label: "Parameter"
    property real value: 0.0
    property real from: 0.0
    property real to: 1.0
    property int precision: 2
    property bool showLabel: true

    // Signals (renamed to avoid conflict with property's auto-signal)
    signal sliderMoved(real newValue)

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        // Top Row: Label + Value
        RowLayout {
            Layout.fillWidth: true
            visible: root.showLabel

            Text {
                text: root.label
                color: Qt.rgba(1, 1, 1, 0.55)
                font.pixelSize: 12
                font.weight: Font.Medium
                font.letterSpacing: 0.3
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                Layout.preferredWidth: 50
                Layout.preferredHeight: 24
                color: Qt.rgba(0, 212, 219, 0.08)
                radius: 6

                Text {
                    anchors.centerIn: parent
                    text: root.value.toFixed(root.precision)
                    color: Theme.accentCyan
                    font.pixelSize: 11
                    font.weight: Font.DemiBold
                    font.family: "Consolas, D2Coding, monospace"
                }
            }
        }

        // Slider
        Slider {
            id: slider
            Layout.fillWidth: true
            from: root.from
            to: root.to
            value: root.value
            stepSize: Math.pow(10, -root.precision)

            onMoved: {
                root.value = value
                root.sliderMoved(value)
            }

            background: Rectangle {
                x: slider.leftPadding
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                width: slider.availableWidth
                height: 5
                radius: 3
                color: Qt.rgba(1, 1, 1, 0.06)

                // Gradient fill for the active portion
                Rectangle {
                    width: slider.visualPosition * parent.width
                    height: parent.height
                    radius: 3

                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: Theme.accentCyan }
                        GradientStop { position: 1.0; color: Theme.accentPurple }
                    }
                }
            }

            handle: Rectangle {
                x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                width: 16
                height: 16
                radius: 8
                color: slider.pressed ? "#00BEC7" : (handleArea.containsMouse ? Theme.accentCyan : "#FFFFFF")

                Behavior on color {
                    ColorAnimation { duration: Theme.animFast }
                }

                Behavior on scale {
                    NumberAnimation { duration: 100; easing.type: Easing.OutQuad }
                }

                scale: slider.pressed ? 1.1 : (handleArea.containsMouse ? 1.05 : 1.0)

                // Hover detection
                MouseArea {
                    id: handleArea
                    anchors.fill: parent
                    anchors.margins: -4
                    hoverEnabled: true
                    acceptedButtons: Qt.NoButton
                }
            }
        }
    }

    // Programmatic value update
    function setValue(newValue) {
        root.value = Math.max(root.from, Math.min(root.to, newValue))
    }
}
