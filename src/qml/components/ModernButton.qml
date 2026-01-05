import QtQuick 2.15
import QtQuick.Controls 2.15
import "../styles"

Button {
    id: root

    property string buttonType: "primary"
    property string iconText: ""
    property bool loading: false

    implicitWidth: Math.max(96, contentText.implicitWidth + 32)
    implicitHeight: Theme.buttonHeightMedium

    contentItem: Row {
        id: contentRow
        spacing: 6
        anchors.centerIn: parent

        // Loading indicator
        Rectangle {
            visible: root.loading
            width: 14
            height: 14
            radius: 7
            color: "transparent"
            border.width: 2
            border.color: "white"
            opacity: 0.8
            anchors.verticalCenter: parent.verticalCenter

            RotationAnimation on rotation {
                running: root.loading
                from: 0
                to: 360
                duration: 1000
                loops: Animation.Infinite
            }

            Rectangle {
                width: 7
                height: 7
                radius: 3.5
                color: root.background.color
                anchors.right: parent.right
                anchors.top: parent.top
            }
        }

        // Icon (optional)
        Text {
            visible: root.iconText !== "" && !root.loading
            text: root.iconText
            font.pixelSize: 14
            color: "#ffffff"
            verticalAlignment: Text.AlignVCenter
            anchors.verticalCenter: parent.verticalCenter
        }

        // Button text
        Text {
            id: contentText
            text: root.text
            color: "#ffffff"
            font.pixelSize: Theme.fontSizeMedium
            font.weight: Font.Medium
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            opacity: root.loading ? 0.5 : 1.0
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    background: Rectangle {
        radius: Theme.radiusSmall
        color: {
            if (root.buttonType === "primary") {
                return root.pressed ? Qt.darker(Theme.blurple, 1.1) :
                       root.hovered ? Theme.blurpleHover : Theme.blurple
            } else if (root.buttonType === "danger") {
                return root.pressed ? Qt.darker(Theme.red, 1.1) :
                       root.hovered ? Theme.redHover : Theme.red
            } else if (root.buttonType === "success") {
                return root.pressed ? Qt.darker(Theme.green, 1.1) :
                       root.hovered ? Qt.darker(Theme.green, 0.9) : Theme.green
            } else if (root.buttonType === "gradient") {
                return "transparent"
            } else {
                return root.pressed ? Theme.bgModifierActive :
                       root.hovered ? Theme.bgModifierHover : Theme.bgModifierSelected
            }
        }

        Behavior on color {
            ColorAnimation { duration: Theme.animFast }
        }

        // Gradient background (only for gradient type)
        gradient: root.buttonType === "gradient" ? gradientBg : null

        Gradient {
            id: gradientBg
            orientation: Gradient.Horizontal
            GradientStop {
                position: 0.0
                color: root.hovered ? Qt.lighter(Theme.blurple, 1.1) : Theme.blurple
            }
            GradientStop {
                position: 1.0
                color: root.hovered ? Qt.lighter("#7B61FF", 1.1) : "#7B61FF"
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        acceptedButtons: Qt.NoButton
    }
}
