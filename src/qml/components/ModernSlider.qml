import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../styles"

Item {
    id: root
    implicitHeight: 48
    implicitWidth: 200

    property string label: "Parameter"
    property real value: 0.0
    property real from: 0.0
    property real to: 1.0
    property int precision: 2
    property bool showPercentage: false
    property bool showLabel: true

    signal sliderMoved(real newValue)

    ColumnLayout {
        anchors.fill: parent
        spacing: Theme.spacingXSmall

        // Top row: Label and Value
        RowLayout {
            Layout.fillWidth: true
            visible: root.showLabel

            Text {
                text: root.label
                color: Theme.textNormal
                font.pixelSize: Theme.fontSizeMedium
                font.weight: Font.Normal
                Layout.fillWidth: true
            }

            Text {
                text: root.showPercentage
                    ? Math.round(root.value * 100) + "%"
                    : root.value.toFixed(root.precision)
                color: Theme.textMuted
                font.pixelSize: Theme.fontSizeSmall
                font.weight: Font.Medium
                horizontalAlignment: Text.AlignRight
            }
        }

        // Slider (full width)
        Slider {
            id: slider
            Layout.fillWidth: true
            Layout.preferredHeight: 20
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
                height: Theme.sliderHeight
                radius: height / 2
                color: Theme.bgModifierActive

                Rectangle {
                    width: slider.visualPosition * parent.width
                    height: parent.height
                    radius: height / 2
                    color: Theme.blurple
                }
            }

            handle: Rectangle {
                x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                width: slider.pressed ? 16 : 14
                height: width
                radius: width / 2
                color: "#ffffff"

                Behavior on width {
                    NumberAnimation { duration: Theme.animFast }
                }
            }
        }
    }

    // Programmatic value update
    function setValue(newValue) {
        root.value = Math.max(root.from, Math.min(root.to, newValue))
    }
}
