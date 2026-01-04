import QtQuick 2.15
import QtQuick.Controls 2.15
import "../styles"

Button {
    id: root

    // Button type: "primary", "secondary", "danger", "gradient"
    property string buttonType: "secondary"
    property string iconText: ""  // Optional icon (emoji or text)
    property bool loading: false

    implicitWidth: Math.max(100, contentRow.implicitWidth + 24)
    implicitHeight: 36

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
            color: "white"
            verticalAlignment: Text.AlignVCenter
        }

        // Button text
        Text {
            text: root.text
            color: "white"
            font.pixelSize: Theme.fontSizeMedium
            font.weight: Font.DemiBold
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            opacity: root.loading ? 0.7 : 1.0
        }
    }

    background: Rectangle {
        id: bgRect
        radius: Theme.radiusMedium
        color: getBackgroundColor()
        border.width: root.buttonType === "secondary" ? 1 : 0
        border.color: Qt.rgba(1, 1, 1, 0.1)

        // Gradient background (only for gradient type)
        gradient: root.buttonType === "gradient" ? gradientBg : null

        Behavior on color {
            ColorAnimation { duration: Theme.animFast }
        }

        // Press scale effect
        transform: Scale {
            origin.x: bgRect.width / 2
            origin.y: bgRect.height / 2
            xScale: root.pressed ? 0.97 : 1.0
            yScale: root.pressed ? 0.97 : 1.0

            Behavior on xScale {
                NumberAnimation { duration: 100; easing.type: Easing.OutQuad }
            }
            Behavior on yScale {
                NumberAnimation { duration: 100; easing.type: Easing.OutQuad }
            }
        }

        Gradient {
            id: gradientBg
            orientation: Gradient.Horizontal
            GradientStop {
                position: 0.0
                color: root.hovered ? Qt.lighter(Theme.accentCyan, 1.1) : Theme.accentCyan
            }
            GradientStop {
                position: 1.0
                color: root.hovered ? Qt.lighter(Theme.accentPurple, 1.1) : Theme.accentPurple
            }
        }
    }

    function getBackgroundColor() {
        if (root.buttonType === "gradient") {
            return "transparent"  // Using gradient instead
        }
        if (root.buttonType === "primary") {
            return root.pressed ? Qt.darker(Theme.accent, 1.1) :
                   root.hovered ? Theme.accentHover : Theme.accent
        }
        if (root.buttonType === "danger") {
            return root.pressed ? Qt.darker(Theme.danger, 1.1) :
                   root.hovered ? Theme.dangerHover : Theme.danger
        }
        // Secondary (default)
        return root.pressed ? Qt.darker(Theme.bgTertiary, 1.1) :
               root.hovered ? Theme.bgHover : Theme.bgTertiary
    }

    // Cursor
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        acceptedButtons: Qt.NoButton
    }
}
