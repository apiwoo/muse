import QtQuick 2.15
import QtQuick.Controls 2.15
import "../styles"

CheckBox {
    id: root

    property string description: ""  // Optional description text

    implicitWidth: contentRow.implicitWidth
    implicitHeight: Math.max(20, contentColumn.implicitHeight)

    indicator: Rectangle {
        id: checkboxBg
        implicitWidth: 18
        implicitHeight: 18
        x: root.leftPadding
        y: parent.height / 2 - height / 2
        radius: 4
        color: root.checked ? Theme.accent : "transparent"
        border.width: root.checked ? 0 : 2
        border.color: root.hovered ? Theme.textPrimary : Theme.textSecondary

        Behavior on color {
            ColorAnimation { duration: Theme.animFast }
        }

        Behavior on border.color {
            ColorAnimation { duration: Theme.animFast }
        }

        // Checkmark
        Item {
            anchors.centerIn: parent
            width: 12
            height: 12
            opacity: root.checked ? 1 : 0
            scale: root.checked ? 1 : 0.5

            Behavior on opacity {
                NumberAnimation { duration: Theme.animFast }
            }

            Behavior on scale {
                NumberAnimation { duration: Theme.animFast; easing.type: Easing.OutBack }
            }

            // Checkmark path using Canvas
            Canvas {
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    ctx.strokeStyle = "white"
                    ctx.lineWidth = 2
                    ctx.lineCap = "round"
                    ctx.lineJoin = "round"
                    ctx.beginPath()
                    ctx.moveTo(2, 6)
                    ctx.lineTo(5, 10)
                    ctx.lineTo(10, 3)
                    ctx.stroke()
                }
            }
        }

        // Hover glow
        Rectangle {
            anchors.fill: parent
            anchors.margins: -3
            radius: 7
            color: Theme.accent
            opacity: root.hovered && !root.checked ? 0.15 : 0

            Behavior on opacity {
                NumberAnimation { duration: Theme.animFast }
            }
        }
    }

    contentItem: Row {
        id: contentRow
        leftPadding: root.indicator.width + 8
        spacing: 0

        Column {
            id: contentColumn
            spacing: 2
            anchors.verticalCenter: parent.verticalCenter

            Text {
                text: root.text
                color: Theme.textPrimary
                font.pixelSize: Theme.fontSizeMedium
                font.weight: Font.Medium
            }

            Text {
                visible: root.description !== ""
                text: root.description
                color: Theme.textSecondary
                font.pixelSize: Theme.fontSizeSmall
                width: Math.min(implicitWidth, 200)
                wrapMode: Text.WordWrap
            }
        }
    }

    // Cursor
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        acceptedButtons: Qt.NoButton
    }
}
